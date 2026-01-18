import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
class Test_as_numeric(unittest.TestCase):

    def test_none(self):
        val = None
        with self.assertRaisesRegex(TypeError, "NoneType values \\('None'\\) are not allowed in Pyomo numeric expressions"):
            as_numeric(val)

    def test_bool(self):
        with self.assertRaisesRegex(TypeError, "bool values \\('False'\\) are not allowed in Pyomo numeric expressions"):
            as_numeric(False)
        with self.assertRaisesRegex(TypeError, "bool values \\('True'\\) are not allowed in Pyomo numeric expressions"):
            as_numeric(True)

    def test_float(self):
        val = 1.1
        nval = as_numeric(val)
        self.assertEqual(val, nval)
        self.assertEqual(nval / 2, 0.55)

    def test_int(self):
        val = 1
        nval = as_numeric(val)
        self.assertEqual(1.0, nval)
        self.assertEqual(nval / 2, 0.5)

    def test_long(self):
        val = int(10000000000.0)
        nval = as_numeric(val)
        self.assertEqual(10000000000.0, nval)
        self.assertEqual(nval / 2, 5000000000.0)

    def test_string(self):
        val = 'foo'
        with self.assertRaisesRegex(TypeError, "str values \\('foo'\\) are not allowed in Pyomo numeric expressions"):
            as_numeric(val)

    def test_const1(self):
        val = NumericConstant(1.0)
        self.assertEqual(1.0, as_numeric(val))

    def test_error1(self):

        class A(object):
            pass
        val = A()
        with self.assertRaisesRegex(TypeError, "Cannot treat the value '.*' as a numeric value because it has unknown type 'A'"):
            as_numeric(val)

    def test_unknownType(self):
        ref = MyBogusType(42)
        with self.assertRaisesRegex(TypeError, "Cannot treat the value '.*' as a numeric value because it has unknown type 'MyBogusType'"):
            as_numeric(ref)

    def test_non_numeric_component(self):
        m = ConcreteModel()
        m.v = Var([1, 2])
        with self.assertRaisesRegex(TypeError, "The 'IndexedVar' object 'v' is not a valid type for Pyomo numeric expressions"):
            as_numeric(m.v)
        obj = PyomoObject()
        with self.assertRaisesRegex(TypeError, "The 'PyomoObject' object '.*' is not a valid type for Pyomo numeric expressions"):
            as_numeric(obj)

    def test_unknownNumericType(self):
        ref = MyBogusNumericType(42)
        self.assertNotIn(MyBogusNumericType, native_numeric_types)
        self.assertNotIn(MyBogusNumericType, native_types)
        try:
            val = as_numeric(ref)
            self.assertEqual(val().val, 42.0)
        finally:
            native_numeric_types.remove(MyBogusNumericType)
            native_types.remove(MyBogusNumericType)

    @unittest.skipUnless(numpy_available, 'This test requires NumPy')
    def test_numpy_basic_float_registration(self):
        self.assertIn(numpy.float_, native_numeric_types)
        self.assertNotIn(numpy.float_, native_integer_types)
        self.assertIn(numpy.float_, _native_boolean_types)
        self.assertIn(numpy.float_, native_types)

    @unittest.skipUnless(numpy_available, 'This test requires NumPy')
    def test_numpy_basic_int_registration(self):
        self.assertIn(numpy.int_, native_numeric_types)
        self.assertIn(numpy.int_, native_integer_types)
        self.assertIn(numpy.int_, _native_boolean_types)
        self.assertIn(numpy.int_, native_types)

    @unittest.skipUnless(numpy_available, 'This test requires NumPy')
    def test_numpy_basic_bool_registration(self):
        self.assertNotIn(numpy.bool_, native_numeric_types)
        self.assertNotIn(numpy.bool_, native_integer_types)
        self.assertIn(numpy.bool_, _native_boolean_types)
        self.assertIn(numpy.bool_, native_types)

    @unittest.skipUnless(numpy_available, 'This test requires NumPy')
    def test_automatic_numpy_registration(self):
        cmd = 'import pyomo; from pyomo.core.base import Var, Param; from pyomo.core.base.units_container import units; import numpy as np; print(np.float64 in pyomo.common.numeric_types.native_numeric_types); %s; print(np.float64 in pyomo.common.numeric_types.native_numeric_types)'

        def _tester(expr):
            rc = subprocess.run([sys.executable, '-c', cmd % expr], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual((rc.returncode, rc.stdout), (0, 'False\nTrue\n'))
        _tester('Var() <= np.float64(5)')
        _tester('np.float64(5) <= Var()')
        _tester('np.float64(5) + Var()')
        _tester('Var() + np.float64(5)')
        _tester('v = Var(); v.construct(); v.value = np.float64(5)')
        _tester('p = Param(mutable=True); p.construct(); p.value = np.float64(5)')
        _tester('v = Var(units=units.m); v.construct(); v.value = np.float64(5)')
        _tester('p = Param(mutable=True, units=units.m); p.construct(); p.value = np.float64(5)')