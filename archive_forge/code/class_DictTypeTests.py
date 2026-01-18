import re
import unittest
from oslo_config import types
class DictTypeTests(TypeTestHelper, unittest.TestCase):
    type = types.Dict()

    def test_empty_value(self):
        self.assertConvertedValue('', {})

    def test_single_value(self):
        self.assertConvertedValue(' foo: bar ', {'foo': 'bar'})

    def test_dict_of_values(self):
        self.assertConvertedValue(' foo: bar, baz: 123 ', {'foo': 'bar', 'baz': '123'})

    def test_custom_value_type(self):
        self.type_instance = types.Dict(types.Integer())
        self.assertConvertedValue('foo:123, bar: 456', {'foo': 123, 'bar': 456})

    def test_dict_of_values_containing_commas(self):
        self.type_instance = types.Dict(types.String(quotes=True))
        self.assertConvertedValue('foo:"bar, baz",bam:quux', {'foo': 'bar, baz', 'bam': 'quux'})

    def test_dict_of_dicts(self):
        self.type_instance = types.Dict(types.Dict(types.String(), bounds=True))
        self.assertConvertedValue('k1:{k1:v1,k2:v2},k2:{k3:v3}', {'k1': {'k1': 'v1', 'k2': 'v2'}, 'k2': {'k3': 'v3'}})

    def test_bounds_parsing(self):
        self.type_instance = types.Dict(types.String(), bounds=True)
        self.assertConvertedValue('{foo:bar,baz:123}', {'foo': 'bar', 'baz': '123'})

    def test_bounds_required(self):
        self.type_instance = types.Dict(types.String(), bounds=True)
        self.assertInvalid('foo:bar,baz:123')
        self.assertInvalid('{foo:bar,baz:123')
        self.assertInvalid('foo:bar,baz:123}')

    def test_no_mapping_produces_error(self):
        self.assertInvalid('foo,bar')

    def test_repr(self):
        t = types.Dict(types.Integer())
        self.assertEqual('Dict of Integer', repr(t))

    def test_equal(self):
        self.assertTrue(types.Dict() == types.Dict())

    def test_equal_with_equal_custom_item_types(self):
        it1 = types.Integer()
        it2 = types.Integer()
        self.assertTrue(types.Dict(it1) == types.Dict(it2))

    def test_not_equal_with_non_equal_custom_item_types(self):
        it1 = types.Integer()
        it2 = types.String()
        self.assertFalse(it1 == it2)
        self.assertFalse(types.Dict(it1) == types.Dict(it2))

    def test_not_equal_to_other_class(self):
        self.assertFalse(types.Dict() == types.Integer())