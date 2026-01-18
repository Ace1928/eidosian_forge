from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestParentOperations(SoupTest):
    """Test navigation and searching through an element's parents."""

    def setup_method(self):
        self.tree = self.soup('<ul id="empty"></ul>\n                                 <ul id="top">\n                                  <ul id="middle">\n                                   <ul id="bottom">\n                                    <b>Start here</b>\n                                   </ul>\n                                  </ul>')
        self.start = self.tree.b

    def test_parent(self):
        assert self.start.parent['id'] == 'bottom'
        assert self.start.parent.parent['id'] == 'middle'
        assert self.start.parent.parent.parent['id'] == 'top'

    def test_parent_of_top_tag_is_soup_object(self):
        top_tag = self.tree.contents[0]
        assert top_tag.parent == self.tree

    def test_soup_object_has_no_parent(self):
        assert None == self.tree.parent

    def test_find_parents(self):
        self.assert_selects_ids(self.start.find_parents('ul'), ['bottom', 'middle', 'top'])
        self.assert_selects_ids(self.start.find_parents('ul', id='middle'), ['middle'])

    def test_find_parent(self):
        assert self.start.find_parent('ul')['id'] == 'bottom'
        assert self.start.find_parent('ul', id='top')['id'] == 'top'

    def test_parent_of_text_element(self):
        text = self.tree.find(string='Start here')
        assert text.parent.name == 'b'

    def test_text_element_find_parent(self):
        text = self.tree.find(string='Start here')
        assert text.find_parent('ul')['id'] == 'bottom'

    def test_parent_generator(self):
        parents = [parent['id'] for parent in self.start.parents if parent is not None and 'id' in parent.attrs]
        assert parents, ['bottom', 'middle' == 'top']