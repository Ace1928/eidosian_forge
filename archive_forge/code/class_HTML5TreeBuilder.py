import warnings
import re
from bs4.builder import (
from bs4.element import (
import html5lib
from html5lib.constants import (
from bs4.element import (
class HTML5TreeBuilder(HTMLTreeBuilder):
    """Use html5lib to build a tree.

    Note that this TreeBuilder does not support some features common
    to HTML TreeBuilders. Some of these features could theoretically
    be implemented, but at the very least it's quite difficult,
    because html5lib moves the parse tree around as it's being built.

    * This TreeBuilder doesn't use different subclasses of NavigableString
      based on the name of the tag in which the string was found.

    * You can't use a SoupStrainer to parse only part of a document.
    """
    NAME = 'html5lib'
    features = [NAME, PERMISSIVE, HTML_5, HTML]
    TRACKS_LINE_NUMBERS = True

    def prepare_markup(self, markup, user_specified_encoding, document_declared_encoding=None, exclude_encodings=None):
        self.user_specified_encoding = user_specified_encoding
        if exclude_encodings:
            warnings.warn("You provided a value for exclude_encoding, but the html5lib tree builder doesn't support exclude_encoding.", stacklevel=3)
        DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(markup, stacklevel=3)
        yield (markup, None, None, False)

    def feed(self, markup):
        if self.soup.parse_only is not None:
            warnings.warn("You provided a value for parse_only, but the html5lib tree builder doesn't support parse_only. The entire document will be parsed.", stacklevel=4)
        parser = html5lib.HTMLParser(tree=self.create_treebuilder)
        self.underlying_builder.parser = parser
        extra_kwargs = dict()
        if not isinstance(markup, str):
            if new_html5lib:
                extra_kwargs['override_encoding'] = self.user_specified_encoding
            else:
                extra_kwargs['encoding'] = self.user_specified_encoding
        doc = parser.parse(markup, **extra_kwargs)
        if isinstance(markup, str):
            doc.original_encoding = None
        else:
            original_encoding = parser.tokenizer.stream.charEncoding[0]
            if not isinstance(original_encoding, str):
                original_encoding = original_encoding.name
            doc.original_encoding = original_encoding
        self.underlying_builder.parser = None

    def create_treebuilder(self, namespaceHTMLElements):
        self.underlying_builder = TreeBuilderForHtml5lib(namespaceHTMLElements, self.soup, store_line_numbers=self.store_line_numbers)
        return self.underlying_builder

    def test_fragment_to_document(self, fragment):
        """See `TreeBuilder`."""
        return '<html><head></head><body>%s</body></html>' % fragment