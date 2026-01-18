from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
class DetectsXMLParsedAsHTML(object):
    """A mixin class for any class (a TreeBuilder, or some class used by a
    TreeBuilder) that's in a position to detect whether an XML
    document is being incorrectly parsed as HTML, and issue an
    appropriate warning.

    This requires being able to observe an incoming processing
    instruction that might be an XML declaration, and also able to
    observe tags as they're opened. If you can't do that for a given
    TreeBuilder, there's a less reliable implementation based on
    examining the raw markup.
    """
    LOOKS_LIKE_HTML = re.compile('<[^ +]html', re.I)
    LOOKS_LIKE_HTML_B = re.compile(b'<[^ +]html', re.I)
    XML_PREFIX = '<?xml'
    XML_PREFIX_B = b'<?xml'

    @classmethod
    def warn_if_markup_looks_like_xml(cls, markup, stacklevel=3):
        """Perform a check on some markup to see if it looks like XML
        that's not XHTML. If so, issue a warning.

        This is much less reliable than doing the check while parsing,
        but some of the tree builders can't do that.

        :param stacklevel: The stacklevel of the code calling this
        function.

        :return: True if the markup looks like non-XHTML XML, False
        otherwise.

        """
        if isinstance(markup, bytes):
            prefix = cls.XML_PREFIX_B
            looks_like_html = cls.LOOKS_LIKE_HTML_B
        else:
            prefix = cls.XML_PREFIX
            looks_like_html = cls.LOOKS_LIKE_HTML
        if markup is not None and markup.startswith(prefix) and (not looks_like_html.search(markup[:500])):
            cls._warn(stacklevel=stacklevel + 2)
            return True
        return False

    @classmethod
    def _warn(cls, stacklevel=5):
        """Issue a warning about XML being parsed as HTML."""
        warnings.warn(XMLParsedAsHTMLWarning.MESSAGE, XMLParsedAsHTMLWarning, stacklevel=stacklevel)

    def _initialize_xml_detector(self):
        """Call this method before parsing a document."""
        self._first_processing_instruction = None
        self._root_tag = None

    def _document_might_be_xml(self, processing_instruction):
        """Call this method when encountering an XML declaration, or a
        "processing instruction" that might be an XML declaration.
        """
        if self._first_processing_instruction is not None or self._root_tag is not None:
            return
        self._first_processing_instruction = processing_instruction

    def _root_tag_encountered(self, name):
        """Call this when you encounter the document's root tag.

        This is where we actually check whether an XML document is
        being incorrectly parsed as HTML, and issue the warning.
        """
        if self._root_tag is not None:
            return
        self._root_tag = name
        if name != 'html' and self._first_processing_instruction is not None and self._first_processing_instruction.lower().startswith('xml '):
            self._warn()