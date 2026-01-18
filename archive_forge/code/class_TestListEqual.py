from unittest import TestCase
from jsonschema._utils import equal
class TestListEqual(TestCase):

    def test_equal_lists(self):
        list_1 = ['a', 'b', 'c']
        list_2 = ['a', 'b', 'c']
        self.assertTrue(equal(list_1, list_2))

    def test_unsorted_lists(self):
        list_1 = ['a', 'b', 'c']
        list_2 = ['b', 'b', 'a']
        self.assertFalse(equal(list_1, list_2))

    def test_first_list_larger(self):
        list_1 = ['a', 'b', 'c']
        list_2 = ['a', 'b']
        self.assertFalse(equal(list_1, list_2))

    def test_second_list_larger(self):
        list_1 = ['a', 'b']
        list_2 = ['a', 'b', 'c']
        self.assertFalse(equal(list_1, list_2))

    def test_list_with_none_unequal(self):
        list_1 = ['a', 'b', None]
        list_2 = ['a', 'b', 'c']
        self.assertFalse(equal(list_1, list_2))
        list_1 = ['a', 'b', None]
        list_2 = [None, 'b', 'c']
        self.assertFalse(equal(list_1, list_2))

    def test_list_with_none_equal(self):
        list_1 = ['a', None, 'c']
        list_2 = ['a', None, 'c']
        self.assertTrue(equal(list_1, list_2))

    def test_empty_list(self):
        list_1 = []
        list_2 = []
        self.assertTrue(equal(list_1, list_2))

    def test_one_none(self):
        list_1 = None
        list_2 = []
        self.assertFalse(equal(list_1, list_2))

    def test_same_list(self):
        list_1 = ['a', 'b', 'c']
        self.assertTrue(equal(list_1, list_1))

    def test_equal_nested_lists(self):
        list_1 = ['a', ['b', 'c'], 'd']
        list_2 = ['a', ['b', 'c'], 'd']
        self.assertTrue(equal(list_1, list_2))

    def test_unequal_nested_lists(self):
        list_1 = ['a', ['b', 'c'], 'd']
        list_2 = ['a', [], 'c']
        self.assertFalse(equal(list_1, list_2))