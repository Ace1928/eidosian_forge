from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestNextSibling(SiblingTest):

    def setup_method(self):
        super(TestNextSibling, self).setup_method()
        self.start = self.tree.find(id='1')

    def test_next_sibling_of_root_is_none(self):
        assert self.tree.next_sibling == None

    def test_next_sibling(self):
        assert self.start.next_sibling['id'] == '2'
        assert self.start.next_sibling.next_sibling['id'] == '3'
        assert self.start.next_element['id'] == '1.1'

    def test_next_sibling_may_not_exist(self):
        assert self.tree.html.next_sibling == None
        nested_span = self.tree.find(id='1.1')
        assert nested_span.next_sibling == None
        last_span = self.tree.find(id='4')
        assert last_span.next_sibling == None

    def test_find_next_sibling(self):
        assert self.start.find_next_sibling('span')['id'] == '2'

    def test_next_siblings(self):
        self.assert_selects_ids(self.start.find_next_siblings('span'), ['2', '3', '4'])
        self.assert_selects_ids(self.start.find_next_siblings(id='3'), ['3'])

    def test_next_sibling_for_text_element(self):
        soup = self.soup('Foo<b>bar</b>baz')
        start = soup.find(string='Foo')
        assert start.next_sibling.name == 'b'
        assert start.next_sibling.next_sibling == 'baz'
        self.assert_selects(start.find_next_siblings('b'), ['bar'])
        assert start.find_next_sibling(string='baz') == 'baz'
        assert start.find_next_sibling(string='nonesuch') == None