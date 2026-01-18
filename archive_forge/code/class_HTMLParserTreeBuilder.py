from html.parser import HTMLParser
import sys
import warnings
from bs4.element import (
from bs4.dammit import EntitySubstitution, UnicodeDammit
from bs4.builder import (
class HTMLParserTreeBuilder(HTMLTreeBuilder):
    """A Beautiful soup `TreeBuilder` that uses the `HTMLParser` parser,
    found in the Python standard library.
    """
    is_xml = False
    picklable = True
    NAME = HTMLPARSER
    features = [NAME, HTML, STRICT]
    TRACKS_LINE_NUMBERS = True

    def __init__(self, parser_args=None, parser_kwargs=None, **kwargs):
        """Constructor.

        :param parser_args: Positional arguments to pass into 
            the BeautifulSoupHTMLParser constructor, once it's
            invoked.
        :param parser_kwargs: Keyword arguments to pass into 
            the BeautifulSoupHTMLParser constructor, once it's
            invoked.
        :param kwargs: Keyword arguments for the superclass constructor.
        """
        extra_parser_kwargs = dict()
        for arg in ('on_duplicate_attribute',):
            if arg in kwargs:
                value = kwargs.pop(arg)
                extra_parser_kwargs[arg] = value
        super(HTMLParserTreeBuilder, self).__init__(**kwargs)
        parser_args = parser_args or []
        parser_kwargs = parser_kwargs or {}
        parser_kwargs.update(extra_parser_kwargs)
        parser_kwargs['convert_charrefs'] = False
        self.parser_args = (parser_args, parser_kwargs)

    def prepare_markup(self, markup, user_specified_encoding=None, document_declared_encoding=None, exclude_encodings=None):
        """Run any preliminary steps necessary to make incoming markup
        acceptable to the parser.

        :param markup: Some markup -- probably a bytestring.
        :param user_specified_encoding: The user asked to try this encoding.
        :param document_declared_encoding: The markup itself claims to be
            in this encoding.
        :param exclude_encodings: The user asked _not_ to try any of
            these encodings.

        :yield: A series of 4-tuples:
         (markup, encoding, declared encoding,
          has undergone character replacement)

         Each 4-tuple represents a strategy for converting the
         document to Unicode and parsing it. Each strategy will be tried 
         in turn.
        """
        if isinstance(markup, str):
            yield (markup, None, None, False)
            return
        known_definite_encodings = [user_specified_encoding]
        user_encodings = [document_declared_encoding]
        try_encodings = [user_specified_encoding, document_declared_encoding]
        dammit = UnicodeDammit(markup, known_definite_encodings=known_definite_encodings, user_encodings=user_encodings, is_html=True, exclude_encodings=exclude_encodings)
        yield (dammit.markup, dammit.original_encoding, dammit.declared_html_encoding, dammit.contains_replacement_characters)

    def feed(self, markup):
        """Run some incoming markup through some parsing process,
        populating the `BeautifulSoup` object in self.soup.
        """
        args, kwargs = self.parser_args
        parser = BeautifulSoupHTMLParser(*args, **kwargs)
        parser.soup = self.soup
        try:
            parser.feed(markup)
            parser.close()
        except AssertionError as e:
            raise ParserRejectedMarkup(e)
        parser.already_closed_empty_element = []