from html.parser import HTMLParser
import sys
import warnings
from bs4.element import (
from bs4.dammit import EntitySubstitution, UnicodeDammit
from bs4.builder import (
class BeautifulSoupHTMLParser(HTMLParser, DetectsXMLParsedAsHTML):
    """A subclass of the Python standard library's HTMLParser class, which
    listens for HTMLParser events and translates them into calls
    to Beautiful Soup's tree construction API.
    """
    IGNORE = 'ignore'
    REPLACE = 'replace'

    def __init__(self, *args, **kwargs):
        """Constructor.

        :param on_duplicate_attribute: A strategy for what to do if a
            tag includes the same attribute more than once. Accepted
            values are: REPLACE (replace earlier values with later
            ones, the default), IGNORE (keep the earliest value
            encountered), or a callable. A callable must take three
            arguments: the dictionary of attributes already processed,
            the name of the duplicate attribute, and the most recent value
            encountered.           
        """
        self.on_duplicate_attribute = kwargs.pop('on_duplicate_attribute', self.REPLACE)
        HTMLParser.__init__(self, *args, **kwargs)
        self.already_closed_empty_element = []
        self._initialize_xml_detector()

    def error(self, message):
        raise ParserRejectedMarkup(message)

    def handle_startendtag(self, name, attrs):
        """Handle an incoming empty-element tag.

        This is only called when the markup looks like <tag/>.

        :param name: Name of the tag.
        :param attrs: Dictionary of the tag's attributes.
        """
        tag = self.handle_starttag(name, attrs, handle_empty_element=False)
        self.handle_endtag(name)

    def handle_starttag(self, name, attrs, handle_empty_element=True):
        """Handle an opening tag, e.g. '<tag>'

        :param name: Name of the tag.
        :param attrs: Dictionary of the tag's attributes.
        :param handle_empty_element: True if this tag is known to be
            an empty-element tag (i.e. there is not expected to be any
            closing tag).
        """
        attr_dict = {}
        for key, value in attrs:
            if value is None:
                value = ''
            if key in attr_dict:
                on_dupe = self.on_duplicate_attribute
                if on_dupe == self.IGNORE:
                    pass
                elif on_dupe in (None, self.REPLACE):
                    attr_dict[key] = value
                else:
                    on_dupe(attr_dict, key, value)
            else:
                attr_dict[key] = value
            attrvalue = '""'
        sourceline, sourcepos = self.getpos()
        tag = self.soup.handle_starttag(name, None, None, attr_dict, sourceline=sourceline, sourcepos=sourcepos)
        if tag and tag.is_empty_element and handle_empty_element:
            self.handle_endtag(name, check_already_closed=False)
            self.already_closed_empty_element.append(name)
        if self._root_tag is None:
            self._root_tag_encountered(name)

    def handle_endtag(self, name, check_already_closed=True):
        """Handle a closing tag, e.g. '</tag>'
        
        :param name: A tag name.
        :param check_already_closed: True if this tag is expected to
           be the closing portion of an empty-element tag,
           e.g. '<tag></tag>'.
        """
        if check_already_closed and name in self.already_closed_empty_element:
            self.already_closed_empty_element.remove(name)
        else:
            self.soup.handle_endtag(name)

    def handle_data(self, data):
        """Handle some textual data that shows up between tags."""
        self.soup.handle_data(data)

    def handle_charref(self, name):
        """Handle a numeric character reference by converting it to the
        corresponding Unicode character and treating it as textual
        data.

        :param name: Character number, possibly in hexadecimal.
        """
        if name.startswith('x'):
            real_name = int(name.lstrip('x'), 16)
        elif name.startswith('X'):
            real_name = int(name.lstrip('X'), 16)
        else:
            real_name = int(name)
        data = None
        if real_name < 256:
            for encoding in (self.soup.original_encoding, 'windows-1252'):
                if not encoding:
                    continue
                try:
                    data = bytearray([real_name]).decode(encoding)
                except UnicodeDecodeError as e:
                    pass
        if not data:
            try:
                data = chr(real_name)
            except (ValueError, OverflowError) as e:
                pass
        data = data or '�'
        self.handle_data(data)

    def handle_entityref(self, name):
        """Handle a named entity reference by converting it to the
        corresponding Unicode character(s) and treating it as textual
        data.

        :param name: Name of the entity reference.
        """
        character = EntitySubstitution.HTML_ENTITY_TO_CHARACTER.get(name)
        if character is not None:
            data = character
        else:
            data = '&%s' % name
        self.handle_data(data)

    def handle_comment(self, data):
        """Handle an HTML comment.

        :param data: The text of the comment.
        """
        self.soup.endData()
        self.soup.handle_data(data)
        self.soup.endData(Comment)

    def handle_decl(self, data):
        """Handle a DOCTYPE declaration.

        :param data: The text of the declaration.
        """
        self.soup.endData()
        data = data[len('DOCTYPE '):]
        self.soup.handle_data(data)
        self.soup.endData(Doctype)

    def unknown_decl(self, data):
        """Handle a declaration of unknown type -- probably a CDATA block.

        :param data: The text of the declaration.
        """
        if data.upper().startswith('CDATA['):
            cls = CData
            data = data[len('CDATA['):]
        else:
            cls = Declaration
        self.soup.endData()
        self.soup.handle_data(data)
        self.soup.endData(cls)

    def handle_pi(self, data):
        """Handle a processing instruction.

        :param data: The text of the instruction.
        """
        self.soup.endData()
        self.soup.handle_data(data)
        self._document_might_be_xml(data)
        self.soup.endData(ProcessingInstruction)