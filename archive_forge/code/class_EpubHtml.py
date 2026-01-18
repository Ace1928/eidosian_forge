import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
class EpubHtml(EpubItem):
    """
    Represents HTML document in the EPUB file.
    """
    _template_name = 'chapter'

    def __init__(self, uid=None, file_name='', media_type='', content=None, title='', lang=None, direction=None, media_overlay=None, media_duration=None):
        super(EpubHtml, self).__init__(uid, file_name, media_type, content)
        self.title = title
        self.lang = lang
        self.direction = direction
        self.media_overlay = media_overlay
        self.media_duration = media_duration
        self.links = []
        self.properties = []
        self.pages = []

    def is_chapter(self):
        """
        Returns if this document is chapter or not.

        :Returns:
          Returns book value.
        """
        return True

    def get_type(self):
        """
        Always returns ebooklib.ITEM_DOCUMENT as type of this document.

        :Returns:
          Always returns ebooklib.ITEM_DOCUMENT
        """
        return ebooklib.ITEM_DOCUMENT

    def set_language(self, lang):
        """
        Sets language for this book item. By default it will use language of the book but it
        can be overwritten with this call.
        """
        self.lang = lang

    def get_language(self):
        """
        Get language code for this book item. Language of the book item can be different from
        the language settings defined globaly for book.

        :Returns:
          As string returns language code.
        """
        return self.lang

    def add_link(self, **kwgs):
        """
        Add additional link to the document. Links will be embeded only inside of this document.

        >>> add_link(href='styles.css', rel='stylesheet', type='text/css')
        """
        self.links.append(kwgs)
        if kwgs.get('type') == 'text/javascript':
            if 'scripted' not in self.properties:
                self.properties.append('scripted')

    def get_links(self):
        """
        Returns list of additional links defined for this document.

        :Returns:
          As tuple return list of links.
        """
        return (link for link in self.links)

    def get_links_of_type(self, link_type):
        """
        Returns list of additional links of specific type.

        :Returns:
          As tuple returns list of links.
        """
        return (link for link in self.links if link.get('type', '') == link_type)

    def add_item(self, item):
        """
        Add other item to this document. It will create additional links according to the item type.

        :Args:
          - item: item we want to add defined as instance of EpubItem
        """
        if item.get_type() == ebooklib.ITEM_STYLE:
            self.add_link(href=item.get_name(), rel='stylesheet', type='text/css')
        if item.get_type() == ebooklib.ITEM_SCRIPT:
            self.add_link(src=item.get_name(), type='text/javascript')

    def get_body_content(self):
        """
        Returns content of BODY element for this HTML document. Content will be of type 'str' (Python 2)
        or 'bytes' (Python 3).

        :Returns:
          Returns content of this document.
        """
        try:
            html_tree = parse_html_string(self.content)
        except:
            return ''
        html_root = html_tree.getroottree()
        if len(html_root.find('body')) != 0:
            body = html_tree.find('body')
            tree_str = etree.tostring(body, pretty_print=True, encoding='utf-8', xml_declaration=False)
            if tree_str.startswith(six.b('<body>')):
                n = tree_str.rindex(six.b('</body>'))
                return tree_str[6:n]
            return tree_str
        return ''

    def get_content(self, default=None):
        """
        Returns content for this document as HTML string. Content will be of type 'str' (Python 2)
        or 'bytes' (Python 3).

        :Args:
          - default: Default value for the content if it is not defined.

        :Returns:
          Returns content of this document.
        """
        tree = parse_string(self.book.get_template(self._template_name))
        tree_root = tree.getroot()
        tree_root.set('lang', self.lang or self.book.language)
        tree_root.attrib['{%s}lang' % NAMESPACES['XML']] = self.lang or self.book.language
        try:
            html_tree = parse_html_string(self.content)
        except:
            return ''
        html_root = html_tree.getroottree()
        _head = etree.SubElement(tree_root, 'head')
        if self.title != '':
            _title = etree.SubElement(_head, 'title')
            _title.text = self.title
        for lnk in self.links:
            if lnk.get('type') == 'text/javascript':
                _lnk = etree.SubElement(_head, 'script', lnk)
                _lnk.text = ''
            else:
                _lnk = etree.SubElement(_head, 'link', lnk)
        _body = etree.SubElement(tree_root, 'body')
        if self.direction:
            _body.set('dir', self.direction)
            tree_root.set('dir', self.direction)
        body = html_tree.find('body')
        if body is not None:
            for i in body.getchildren():
                _body.append(i)
        tree_str = etree.tostring(tree, pretty_print=True, encoding='utf-8', xml_declaration=True)
        return tree_str

    def __str__(self):
        return '<EpubHtml:%s:%s>' % (self.id, self.file_name)