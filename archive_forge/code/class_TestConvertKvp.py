from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertKvp(base.BaseTestCase):

    def test_convert_kvp_list_to_dict_succeeds_for_missing_values(self):
        result = converters.convert_kvp_list_to_dict(['True'])
        self.assertEqual({}, result)

    def test_convert_kvp_list_to_dict_succeeds_for_multiple_values(self):
        result = converters.convert_kvp_list_to_dict(['a=b', 'a=c', 'a=c', 'b=a'])
        expected = {'a': tools.UnorderedList(['c', 'b']), 'b': ['a']}
        self.assertEqual(expected, result)

    def test_convert_kvp_list_to_dict_succeeds_for_values(self):
        result = converters.convert_kvp_list_to_dict(['a=b', 'c=d'])
        self.assertEqual({'a': ['b'], 'c': ['d']}, result)

    def test_convert_kvp_str_to_list_fails_for_missing_key(self):
        with testtools.ExpectedException(n_exc.InvalidInput):
            converters.convert_kvp_str_to_list('=a')

    def test_convert_kvp_str_to_list_fails_for_missing_equals(self):
        with testtools.ExpectedException(n_exc.InvalidInput):
            converters.convert_kvp_str_to_list('a')

    def test_convert_kvp_str_to_list_succeeds_for_one_equals(self):
        result = converters.convert_kvp_str_to_list('a=')
        self.assertEqual(['a', ''], result)

    def test_convert_kvp_str_to_list_succeeds_for_two_equals(self):
        result = converters.convert_kvp_str_to_list('a=a=a')
        self.assertEqual(['a', 'a=a'], result)