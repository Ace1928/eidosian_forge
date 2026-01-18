from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestFind(SoupTest):
    """Basic tests of the find() method.

    find() just calls find_all() with limit=1, so it's not tested all
    that thouroughly here.
    """

    def test_find_tag(self):
        soup = self.soup('<a>1</a><b>2</b><a>3</a><b>4</b>')
        assert soup.find('b').string == '2'

    def test_unicode_text_find(self):
        soup = self.soup('<h1>Räksmörgås</h1>')
        assert soup.find(string='Räksmörgås') == 'Räksmörgås'

    def test_unicode_attribute_find(self):
        soup = self.soup('<h1 id="Räksmörgås">here it is</h1>')
        str(soup)
        assert 'here it is' == soup.find(id='Räksmörgås').text

    def test_find_everything(self):
        """Test an optimization that finds all tags."""
        soup = self.soup('<a>foo</a><b>bar</b>')
        assert 2 == len(soup.find_all())

    def test_find_everything_with_name(self):
        """Test an optimization that finds all tags with a given name."""
        soup = self.soup('<a>foo</a><b>bar</b><a>baz</a>')
        assert 2 == len(soup.find_all('a'))