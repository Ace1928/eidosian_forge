from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestFindAllByName(SoupTest):
    """Test ways of finding tags by tag name."""

    def setup_method(self):
        self.tree = self.soup('<a>First tag.</a>\n                                  <b>Second tag.</b>\n                                  <c>Third <a>Nested tag.</a> tag.</c>')

    def test_find_all_by_tag_name(self):
        self.assert_selects(self.tree.find_all('a'), ['First tag.', 'Nested tag.'])

    def test_find_all_by_name_and_text(self):
        self.assert_selects(self.tree.find_all('a', string='First tag.'), ['First tag.'])
        self.assert_selects(self.tree.find_all('a', string=True), ['First tag.', 'Nested tag.'])
        self.assert_selects(self.tree.find_all('a', string=re.compile('tag')), ['First tag.', 'Nested tag.'])

    def test_find_all_on_non_root_element(self):
        self.assert_selects(self.tree.c.find_all('a'), ['Nested tag.'])

    def test_calling_element_invokes_find_all(self):
        self.assert_selects(self.tree('a'), ['First tag.', 'Nested tag.'])

    def test_find_all_by_tag_strainer(self):
        self.assert_selects(self.tree.find_all(SoupStrainer('a')), ['First tag.', 'Nested tag.'])

    def test_find_all_by_tag_names(self):
        self.assert_selects(self.tree.find_all(['a', 'b']), ['First tag.', 'Second tag.', 'Nested tag.'])

    def test_find_all_by_tag_dict(self):
        self.assert_selects(self.tree.find_all({'a': True, 'b': True}), ['First tag.', 'Second tag.', 'Nested tag.'])

    def test_find_all_by_tag_re(self):
        self.assert_selects(self.tree.find_all(re.compile('^[ab]$')), ['First tag.', 'Second tag.', 'Nested tag.'])

    def test_find_all_with_tags_matching_method(self):

        def id_matches_name(tag):
            return tag.name == tag.get('id')
        tree = self.soup('<a id="a">Match 1.</a>\n                            <a id="1">Does not match.</a>\n                            <b id="b">Match 2.</a>')
        self.assert_selects(tree.find_all(id_matches_name), ['Match 1.', 'Match 2.'])

    def test_find_with_multi_valued_attribute(self):
        soup = self.soup("<div class='a b'>1</div><div class='a c'>2</div><div class='a d'>3</div>")
        r1 = soup.find('div', 'a d')
        r2 = soup.find('div', re.compile('a d'))
        r3, r4 = soup.find_all('div', ['a b', 'a d'])
        assert '3' == r1.string
        assert '3' == r2.string
        assert '1' == r3.string
        assert '3' == r4.string