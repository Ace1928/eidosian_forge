import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
class Test_is_fixed(unittest.TestCase):

    def test_none(self):
        self.assertTrue(is_fixed(None))

    def test_bool(self):
        self.assertTrue(is_fixed(True))

    def test_float(self):
        self.assertTrue(is_fixed(1.1))

    def test_int(self):
        self.assertTrue(is_fixed(1))

    def test_long(self):
        val = int(10000000000.0)
        self.assertTrue(is_fixed(val))

    def test_string(self):
        self.assertTrue(is_fixed('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertTrue(is_fixed(val))

    def test_error(self):

        class A(object):
            pass
        val = A()
        with self.assertRaisesRegex(TypeError, 'Cannot assess properties of object with unknown type: A'):
            is_fixed(val)

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertTrue(is_fixed(ref))
        self.assertIn(MyBogusNumericType, native_numeric_types)
        self.assertIn(MyBogusNumericType, native_types)
        native_numeric_types.remove(MyBogusNumericType)
        native_types.remove(MyBogusNumericType)