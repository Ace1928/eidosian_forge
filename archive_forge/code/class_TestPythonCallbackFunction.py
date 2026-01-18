import os
import shutil
import sys
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.gsl import find_GSL
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.core.base.external import (
from pyomo.core.base.units_container import pint_available, units
from pyomo.core.expr.numeric_expr import (
from pyomo.opt import check_available_solvers
class TestPythonCallbackFunction(unittest.TestCase):

    def test_constructor_errors(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(ValueError, "Duplicate definition of external function through positional and keyword \\('function='\\)"):
            m.f = ExternalFunction(_count, function=_count)
        with self.assertRaisesRegex(ValueError, 'PythonCallbackFunction constructor only supports 0 - 3 positional arguments'):
            m.f = ExternalFunction(1, 2, 3, 4)
        with self.assertRaisesRegex(ValueError, "Cannot specify 'fgh' with any of {'function', 'gradient', hessian'}"):
            m.f = ExternalFunction(_count, fgh=_fgh)

    def test_call_countArgs(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_count)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(value(m.f()), 0)
        self.assertEqual(value(m.f(2)), 1)
        self.assertEqual(value(m.f(2, 3)), 2)

    def test_call_sumfcn(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(value(m.f()), 2.0)
        self.assertEqual(value(m.f(1)), 3.0)
        self.assertEqual(value(m.f(1, 2)), 5.0)

    def test_evaluate_fgh_fgh(self):
        m = ConcreteModel()
        m.f = ExternalFunction(fgh=_fgh)
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 3 * 5 + 5 * 11 ** 2, 2 * 5 * 7 * 11, 0])
        self.assertEqual(h, [2, 3 + 11 ** 2, 0, 2 * 7 * 11, 2 * 5 * 11, 2 * 5 * 7, 0, 0, 0, 0])
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fixed=[0, 1, 0, 1])
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 0, 2 * 5 * 7 * 11, 0])
        self.assertEqual(h, [2, 0, 0, 2 * 7 * 11, 0, 2 * 5 * 7, 0, 0, 0, 0])

    def test_evaluate_fgh_f_g_h(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_f, _g, _h)
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 3 * 5 + 5 * 11 ** 2, 2 * 5 * 7 * 11, 0])
        self.assertEqual(h, [2, 3 + 11 ** 2, 0, 2 * 7 * 11, 2 * 5 * 11, 2 * 5 * 7, 0, 0, 0, 0])
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fixed=[0, 1, 0, 1])
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 0, 2 * 5 * 7 * 11, 0])
        self.assertEqual(h, [2, 0, 0, 2 * 7 * 11, 0, 2 * 5 * 7, 0, 0, 0, 0])
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 3 * 5 + 5 * 11 ** 2, 2 * 5 * 7 * 11, 0])
        self.assertIsNone(h)
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=0)
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertIsNone(g)
        self.assertIsNone(h)

    def test_evaluate_fgh_f_g(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_f, _g)
        with self.assertRaisesRegex(RuntimeError, "ExternalFunction 'f' was not defined with a Hessian callback."):
            f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertEqual(g, [2 * 5 + 3 * 7 + 7 * 11 ** 2, 3 * 5 + 5 * 11 ** 2, 2 * 5 * 7 * 11, 0])
        self.assertIsNone(h)
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=0)
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertIsNone(g)
        self.assertIsNone(h)

    def test_evaluate_fgh_f(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_f)
        with self.assertRaisesRegex(RuntimeError, "ExternalFunction 'f' was not defined with a gradient callback."):
            f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id))
        with self.assertRaisesRegex(RuntimeError, "ExternalFunction 'f' was not defined with a gradient callback."):
            f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
        f, g, h = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=0)
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        self.assertIsNone(g)
        self.assertIsNone(h)

    def test_evaluate_errors(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_f, _g_bad, _h_bad)
        f = m.f.evaluate((5, 7, 11, m.f._fcn_id))
        self.assertEqual(f, 5 ** 2 + 3 * 5 * 7 + 5 * 7 * 11 ** 2)
        with self.assertRaisesRegex(RuntimeError, 'PythonCallbackFunction called with invalid Global ID'):
            f = m.f.evaluate((5, 7, 11, -1))
        with self.assertRaisesRegex(RuntimeError, "External function 'f' returned an invalid derivative vector \\(expected 4, received 5\\)"):
            f = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=1)
        with self.assertRaisesRegex(RuntimeError, "External function 'f' returned an invalid Hessian matrix \\(expected 10, received 9\\)"):
            f = m.f.evaluate_fgh((5, 7, 11, m.f._fcn_id), fgh=2)

    def test_getname(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f.name, 'f')
        self.assertEqual(m.f.local_name, 'f')
        self.assertEqual(m.f.getname(), 'f')
        self.assertEqual(m.f.getname(True), 'f')
        M = ConcreteModel()
        M.m = m
        self.assertEqual(M.m.f.name, 'm.f')
        self.assertEqual(M.m.f.local_name, 'f')
        self.assertEqual(M.m.f.getname(), 'f')
        self.assertEqual(M.m.f.getname(True), 'm.f')

    def test_extra_kwargs(self):
        m = ConcreteModel()
        with self.assertRaises(ValueError):
            m.f = ExternalFunction(_count, this_should_raise_error='foo')

    def test_clone(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        m.x = Var(initialize=3)
        m.y = Var(initialize=5)
        m.e = Expression(expr=m.f(m.x, m.y))
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f._fcn_id, m.e.arg(0).arg(-1).value)
        self.assertEqual(value(m.e), 10)
        i = m.clone()
        self.assertIsInstance(i.f, PythonCallbackFunction)
        self.assertIsNot(i.f, m.f)
        self.assertIsNot(i.e, m.e)
        self.assertIsNot(i.e.arg(0), m.e.arg(0))
        self.assertEqual(i.f._fcn_id, i.e.arg(0).arg(-1).value)
        self.assertNotEqual(i.f._fcn_id, m.f._fcn_id)
        self.assertEqual(value(i.e), 10)

    def test_partial_clone(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        m.x = Var(initialize=3)
        m.y = Var(initialize=5)
        m.b = Block()
        m.b.e = Expression(expr=m.f(m.x, m.y))
        self.assertIsInstance(m.f, PythonCallbackFunction)
        self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
        self.assertEqual(value(m.b.e), 10)
        m.c = m.b.clone()
        self.assertIsNot(m.b.e, m.c.e)
        self.assertIsNot(m.b.e.arg(0), m.c.e.arg(0))
        self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
        self.assertEqual(m.f._fcn_id, m.c.e.arg(0).arg(-1).value)
        self.assertEqual(value(m.c.e), 10)
        _fcn_id = m.f._fcn_id
        m.f.__setstate__(m.f.__getstate__())
        self.assertEqual(m.f._fcn_id, _fcn_id)
        self.assertIsNot(m.b.e, m.c.e)
        self.assertIsNot(m.b.e.arg(0), m.c.e.arg(0))
        self.assertEqual(m.f._fcn_id, m.b.e.arg(0).arg(-1).value)
        self.assertEqual(m.f._fcn_id, m.c.e.arg(0).arg(-1).value)
        self.assertEqual(value(m.c.e), 10)

    def test_properties(self):
        m = ConcreteModel()
        m.f = ExternalFunction(_sum)
        e = m.f()
        self.assertIsInstance(e, NPV_ExternalFunctionExpression)
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertFalse(e.is_potentially_variable())
        self.assertFalse(e.arg(0).is_constant())
        self.assertTrue(e.arg(0).is_fixed())
        self.assertFalse(e.arg(0).is_potentially_variable())
        m.p = Param(initialize=1)
        e = m.f(m.p)
        self.assertIsInstance(e, NPV_ExternalFunctionExpression)
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        self.assertFalse(e.is_potentially_variable())
        m.x = Var(initialize=1)
        e = m.f(m.p, m.x)
        self.assertIsInstance(e, ExternalFunctionExpression)
        self.assertFalse(e.is_constant())
        self.assertFalse(e.is_fixed())
        self.assertTrue(e.is_potentially_variable())

    def test_pprint(self):
        m = ConcreteModel()
        m.h = ExternalFunction(_count)
        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(out.getvalue().strip(), '\n1 ExternalFunction Declarations\n    h : function=_count, units=None, arg_units=None\n\n1 Declarations: h\n        '.strip())
        if not pint_available:
            return
        m.i = ExternalFunction(function=_sum, units=units.kg, arg_units=[units.m, units.s])
        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(out.getvalue().strip(), "\n2 ExternalFunction Declarations\n    h : function=_count, units=None, arg_units=None\n    i : function=_sum, units=kg, arg_units=['m', 's']\n\n2 Declarations: h i\n        ".strip())

    def test_pprint(self):
        m = ConcreteModel()
        m.h = ExternalFunction(_g)
        out = StringIO()
        m.pprint()
        m.pprint(ostream=out)
        self.assertEqual(out.getvalue().strip(), '\n1 ExternalFunction Declarations\n    h : function=_g, units=None, arg_units=None\n\n1 Declarations: h\n        '.strip())
        if not pint_available:
            return
        m.i = ExternalFunction(function=_h, units=units.kg, arg_units=[units.m, units.s])
        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(out.getvalue().strip(), "\n2 ExternalFunction Declarations\n    h : function=_g, units=None, arg_units=None\n    i : function=_h, units=kg, arg_units=['m', 's']\n\n2 Declarations: h i\n        ".strip())