import datetime
import unittest
from wsme.api import FunctionArgument, FunctionDefinition
from wsme.rest.args import from_param, from_params, args_from_args
from wsme.exc import InvalidInput
from wsme.types import UserType, Unset, ArrayType, DictType, Base
class ArgTypeConversion(unittest.TestCase):

    def test_int_zero(self):
        self.assertEqual(0, from_param(int, 0))
        self.assertEqual(0, from_param(int, '0'))

    def test_int_nonzero(self):
        self.assertEqual(1, from_param(int, 1))
        self.assertEqual(1, from_param(int, '1'))

    def test_int_none(self):
        self.assertEqual(None, from_param(int, None))

    def test_float_zero(self):
        self.assertEqual(0.0, from_param(float, 0))
        self.assertEqual(0.0, from_param(float, 0.0))
        self.assertEqual(0.0, from_param(float, '0'))
        self.assertEqual(0.0, from_param(float, '0.0'))

    def test_float_nonzero(self):
        self.assertEqual(1.0, from_param(float, 1))
        self.assertEqual(1.0, from_param(float, 1.0))
        self.assertEqual(1.0, from_param(float, '1'))
        self.assertEqual(1.0, from_param(float, '1.0'))

    def test_float_none(self):
        self.assertEqual(None, from_param(float, None))