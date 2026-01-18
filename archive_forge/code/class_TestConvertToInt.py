from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertToInt(base.BaseTestCase):

    def test_convert_to_int_int(self):
        self.assertEqual(-1, converters.convert_to_int(-1))
        self.assertEqual(0, converters.convert_to_int(0))
        self.assertEqual(1, converters.convert_to_int(1))

    def test_convert_to_int_if_not_none(self):
        self.assertEqual(-1, converters.convert_to_int_if_not_none(-1))
        self.assertEqual(0, converters.convert_to_int_if_not_none(0))
        self.assertEqual(1, converters.convert_to_int_if_not_none(1))
        self.assertIsNone(converters.convert_to_int_if_not_none(None))

    def test_convert_to_int_str(self):
        self.assertEqual(4, converters.convert_to_int('4'))
        self.assertEqual(6, converters.convert_to_int('6'))
        self.assertRaises(n_exc.InvalidInput, converters.convert_to_int, 'garbage')

    def test_convert_to_int_none(self):
        self.assertRaises(n_exc.InvalidInput, converters.convert_to_int, None)

    def test_convert_none_to_empty_list_none(self):
        self.assertEqual([], converters.convert_none_to_empty_list(None))

    def test_convert_none_to_empty_dict(self):
        self.assertEqual({}, converters.convert_none_to_empty_dict(None))

    def test_convert_none_to_empty_list_value(self):
        values = ['1', 3, [], [1], {}, {'a': 3}]
        for value in values:
            self.assertEqual(value, converters.convert_none_to_empty_list(value))