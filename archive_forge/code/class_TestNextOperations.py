from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
from bs4.element import (
from . import (
class TestNextOperations(ProximityTest):

    def setup_method(self):
        super(TestNextOperations, self).setup_method()
        self.start = self.tree.b

    def test_next(self):
        assert self.start.next_element == 'One'
        assert self.start.next_element.next_element['id'] == '2'

    def test_next_of_last_item_is_none(self):
        last = self.tree.find(string='Three')
        assert last.next_element == None

    def test_next_of_root_is_none(self):
        assert self.tree.next_element == None

    def test_find_all_next(self):
        self.assert_selects(self.start.find_all_next('b'), ['Two', 'Three'])
        self.start.find_all_next(id=3)
        self.assert_selects(self.start.find_all_next(id=3), ['Three'])

    def test_find_next(self):
        assert self.start.find_next('b')['id'] == '2'
        assert self.start.find_next(string='Three') == 'Three'

    def test_find_next_for_text_element(self):
        text = self.tree.find(string='One')
        assert text.find_next('b').string == 'Two'
        self.assert_selects(text.find_all_next('b'), ['Two', 'Three'])

    def test_next_generator(self):
        start = self.tree.find(string='Two')
        successors = [node for node in start.next_elements]
        tag, contents = successors
        assert tag['id'] == '3'
        assert contents == 'Three'