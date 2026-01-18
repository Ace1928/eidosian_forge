import re
import unittest
from oslo_config import types
class ListTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.List()

    def test_empty_value(self):
        self.assertConvertedValue('', [])

    def test_single_value(self):
        self.assertConvertedValue(' foo bar ', ['foo bar'])

    def test_tuple_of_values(self):
        self.assertConvertedValue(('foo', 'bar'), ['foo', 'bar'])

    def test_list_of_values(self):
        self.assertConvertedValue(' foo bar, baz ', ['foo bar', 'baz'])

    def test_list_of_values_containing_commas(self):
        self.type_instance = types.List(types.String(quotes=True))
        self.assertConvertedValue('foo,"bar, baz",bam', ['foo', 'bar, baz', 'bam'])

    def test_list_of_values_containing_trailing_comma(self):
        self.assertConvertedValue('foo, bar, baz, ', ['foo', 'bar', 'baz'])

    def test_list_of_lists(self):
        self.type_instance = types.List(types.List(types.String(), bounds=True))
        self.assertConvertedValue('[foo],[bar, baz],[bam]', [['foo'], ['bar', 'baz'], ['bam']])

    def test_list_of_custom_type(self):
        self.type_instance = types.List(types.Integer())
        self.assertConvertedValue('1,2,3,5', [1, 2, 3, 5])

    def test_list_of_custom_type_containing_trailing_comma(self):
        self.type_instance = types.List(types.Integer())
        self.assertConvertedValue('1,2,3,5,', [1, 2, 3, 5])

    def test_tuple_of_custom_type(self):
        self.type_instance = types.List(types.Integer())
        self.assertConvertedValue(('1', '2', '3', '5'), [1, 2, 3, 5])

    def test_bounds_parsing(self):
        self.type_instance = types.List(types.Integer(), bounds=True)
        self.assertConvertedValue('[1,2,3]', [1, 2, 3])

    def test_bounds_required(self):
        self.type_instance = types.List(types.Integer(), bounds=True)
        self.assertInvalid('1,2,3')
        self.assertInvalid('[1,2,3')
        self.assertInvalid('1,2,3]')

    def test_repr(self):
        t = types.List(types.Integer())
        self.assertEqual('List of Integer', repr(t))

    def test_equal(self):
        self.assertTrue(types.List() == types.List())

    def test_equal_with_equal_custom_item_types(self):
        it1 = types.Integer()
        it2 = types.Integer()
        self.assertTrue(types.List(it1) == types.List(it2))

    def test_not_equal_with_non_equal_custom_item_types(self):
        it1 = types.Integer()
        it2 = types.String()
        self.assertFalse(it1 == it2)
        self.assertFalse(types.List(it1) == types.List(it2))

    def test_not_equal_to_other_class(self):
        self.assertFalse(types.List() == types.Integer())