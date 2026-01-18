import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
class Test_is_potentially_variable(unittest.TestCase):

    def test_none(self):
        self.assertFalse(is_potentially_variable(None))

    def test_bool(self):
        self.assertFalse(is_potentially_variable(True))

    def test_float(self):
        self.assertFalse(is_potentially_variable(1.1))

    def test_int(self):
        self.assertFalse(is_potentially_variable(1))

    def test_long(self):
        val = int(10000000000.0)
        self.assertFalse(is_potentially_variable(val))

    def test_string(self):
        self.assertFalse(is_potentially_variable('foo'))

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertFalse(is_potentially_variable(val))

    def test_error(self):

        class A(object):
            pass
        val = A()
        self.assertFalse(is_potentially_variable(val))

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertFalse(is_potentially_variable(ref))