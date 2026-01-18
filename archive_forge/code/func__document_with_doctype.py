import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def _document_with_doctype(self, doctype_fragment, doctype_string='DOCTYPE'):
    """Generate and parse a document with the given doctype."""
    doctype = '<!%s %s>' % (doctype_string, doctype_fragment)
    markup = doctype + '\n<p>foo</p>'
    soup = self.soup(markup)
    return (doctype.encode('utf8'), soup)