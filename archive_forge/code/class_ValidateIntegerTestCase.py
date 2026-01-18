import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
@ddt.ddt
class ValidateIntegerTestCase(test_base.BaseTestCase):

    @ddt.unpack
    @ddt.data({'value': 42, 'name': 'answer', 'output': 42}, {'value': '42', 'name': 'answer', 'output': 42}, {'value': '7', 'name': 'lucky', 'output': 7, 'min_value': 7, 'max_value': 8}, {'value': 7, 'name': 'lucky', 'output': 7, 'min_value': 6, 'max_value': 7}, {'value': 300, 'name': 'Spartaaa!!!', 'output': 300, 'min_value': 300}, {'value': '300', 'name': 'Spartaaa!!!', 'output': 300, 'max_value': 300})
    def test_valid_inputs(self, output, value, name, **kwargs):
        self.assertEqual(strutils.validate_integer(value, name, **kwargs), output)

    @ddt.unpack
    @ddt.data({'value': 'im-not-an-int', 'name': ''}, {'value': 3.14, 'name': 'Pie'}, {'value': '299', 'name': 'Sparta no-show', 'min_value': 300, 'max_value': 300}, {'value': 55, 'name': 'doing 55 in a 54', 'max_value': 54}, {'value': chr(129), 'name': 'UnicodeError', 'max_value': 1000})
    def test_invalid_inputs(self, value, name, **kwargs):
        self.assertRaises(ValueError, strutils.validate_integer, value, name, **kwargs)