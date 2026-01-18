import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
class MiscIndexedParamBehaviorTests(unittest.TestCase):

    def test_mutable_self1(self):
        model = ConcreteModel()
        model.P = Param([1], mutable=True)
        model.P[1] = 1.0
        model.x = Var()
        model.CON = Constraint(expr=model.P[1] <= model.x)
        self.assertEqual(1.0, value(model.CON[None].lower))
        model.P[1] = 2.0
        self.assertEqual(2.0, value(model.CON[None].lower))

    def test_mutable_self2(self):
        model = ConcreteModel()
        model.P = Param([1], initialize=1.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.P[1] <= model.x)
        self.assertEqual(1.0, value(model.CON[None].lower))
        model.P[1] = 2.0
        self.assertEqual(2.0, value(model.CON[None].lower))

    def test_mutable_self3(self):
        model = ConcreteModel()
        model.P = Param([1], default=1.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.P[1] <= model.x)
        self.assertEqual(1.0, value(model.CON[None].lower))
        model.P[1] = 2.0
        self.assertEqual(2.0, value(model.CON[None].lower))

    def test_mutable_self4(self):
        model = ConcreteModel()
        model.P = Param([1, 2], default=1.0, mutable=True)
        self.assertEqual(model.P[1].value, 1.0)
        self.assertEqual(model.P[2].value, 1.0)
        model.P[1].value = 0.0
        self.assertEqual(model.P[1].value, 0.0)
        self.assertEqual(model.P[2].value, 1.0)
        model.Q = Param([1, 2], default=1.0, mutable=True)
        self.assertEqual(model.Q[1].value, 1.0)
        self.assertEqual(model.Q[2].value, 1.0)
        model.Q[1] = 0.0
        self.assertEqual(model.Q[1].value, 0.0)
        self.assertEqual(model.Q[2].value, 1.0)

    def test_mutable_display(self):
        model = ConcreteModel()
        model.P = Param([1, 2], default=0.0, mutable=True)
        model.Q = Param([1, 2], initialize=0.0, mutable=True)
        model.R = Param([1, 2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        for Item in [model.P]:
            f = StringIO()
            display(Item, f)
            tmp = f.getvalue().splitlines()
            self.assertEqual(len(tmp), 2)
        for Item in [model.Q, model.R]:
            f = StringIO()
            display(Item, f)
            tmp = f.getvalue().splitlines()
            for tmp_ in tmp[2:]:
                val = float(tmp_.split(':')[-1].strip())
                self.assertEqual(0, val)
        for Item in [model.P, model.Q, model.R]:
            for i in [1, 2]:
                self.assertEqual(Item[i].value, 0.0)
        for Item in [model.P, model.Q, model.R]:
            f = StringIO()
            display(Item, f)
            tmp = f.getvalue().splitlines()
            for tmp_ in tmp[2:]:
                val = float(tmp_.split(':')[-1].strip())
                self.assertEqual(0, val)
        model.P[1] = 1.0
        model.P[2] = 2.0
        model.Q[1] = 1.0
        model.Q[2] = 2.0
        model.R[1] = 1.0
        model.R[2] = 2.0
        for Item in [model.P, model.Q, model.R]:
            f = StringIO()
            display(Item, f)
            tmp = f.getvalue().splitlines()
            i = 0
            for tmp_ in tmp[2:]:
                i += 1
                val = float(tmp_.split(':')[-1].strip())
                self.assertEqual(i, val)

    def test_mutable_pprint(self):
        model = ConcreteModel()
        model.P = Param([1, 2], default=0.0, mutable=True)
        model.Q = Param([1, 2], initialize=0.0, mutable=True)
        model.R = Param([1, 2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        for Item in [model.P]:
            f = StringIO()
            display(Item, f)
            tmp = f.getvalue().splitlines()
            self.assertEqual(len(tmp), 2)
        for Item in [model.Q, model.R]:
            f = StringIO()
            display(Item, f)
            tmp = f.getvalue().splitlines()
            for tmp_ in tmp[2:]:
                val = float(tmp_.split(':')[-1].strip())
                self.assertEqual(0, val)
        for Item in [model.P, model.Q, model.R]:
            for i in [1, 2]:
                self.assertEqual(Item[i].value, 0.0)
        for Item in [model.P, model.Q, model.R]:
            f = StringIO()
            Item.pprint(ostream=f)
            tmp = f.getvalue().splitlines()
            for i in [1, 2]:
                val = float(tmp[i + 1].split(':')[-1].strip())
                self.assertEqual(0, val)
        model.P[1] = 1.0
        model.P[2] = 2.0
        model.Q[1] = 1.0
        model.Q[2] = 2.0
        model.R[1] = 1.0
        model.R[2] = 2.0
        for Item in [model.P, model.Q, model.R]:
            f = StringIO()
            Item.pprint(ostream=f)
            tmp = f.getvalue().splitlines()
            for i in [1, 2]:
                val = float(tmp[i + 1].split(':')[-1].strip())
                self.assertEqual(i, val)

    def test_mutable_sum_expr(self):
        model = ConcreteModel()
        model.P = Param([1, 2], default=0.0, mutable=True)
        model.Q = Param([1, 2], initialize=0.0, mutable=True)
        model.R = Param([1, 2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        model.x = Var()
        model.CON1 = Constraint(expr=model.P[1] + model.P[2] <= model.x)
        model.CON2 = Constraint(expr=model.Q[1] + model.Q[2] <= model.x)
        model.CON3 = Constraint(expr=model.R[1] + model.R[2] <= model.x)
        self.assertEqual(0.0, value(model.CON1[None].lower))
        self.assertEqual(0.0, value(model.CON2[None].lower))
        self.assertEqual(0.0, value(model.CON3[None].lower))
        model.P[1] = 3.0
        model.P[2] = 2.0
        model.Q[1] = 3.0
        model.Q[2] = 2.0
        model.R[1] = 3.0
        model.R[2] = 2.0
        self.assertEqual(5.0, value(model.CON1[None].lower))
        self.assertEqual(5.0, value(model.CON2[None].lower))
        self.assertEqual(5.0, value(model.CON3[None].lower))

    def test_mutable_prod_expr(self):
        model = ConcreteModel()
        model.P = Param([1, 2], initialize=0.0, mutable=True)
        model.Q = Param([1, 2], default=0.0, mutable=True)
        model.R = Param([1, 2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        model.x = Var()
        model.CON1 = Constraint(expr=model.P[1] * model.P[2] <= model.x)
        model.CON2 = Constraint(expr=model.Q[1] * model.Q[2] <= model.x)
        model.CON3 = Constraint(expr=model.R[1] * model.R[2] <= model.x)
        self.assertEqual(0.0, value(model.CON1[None].lower))
        self.assertEqual(0.0, value(model.CON2[None].lower))
        self.assertEqual(0.0, value(model.CON3[None].lower))
        model.P[1] = 3.0
        model.P[2] = 2.0
        model.Q[1] = 3.0
        model.Q[2] = 2.0
        model.R[1] = 3.0
        model.R[2] = 2.0
        self.assertEqual(6.0, value(model.CON1[None].lower))
        self.assertEqual(6.0, value(model.CON2[None].lower))
        self.assertEqual(6.0, value(model.CON3[None].lower))

    def test_mutable_pow_expr(self):
        model = ConcreteModel()
        model.P = Param([1, 2], initialize=0.0, mutable=True)
        model.Q = Param([1, 2], default=0.0, mutable=True)
        model.R = Param([1, 2], mutable=True)
        model.R[1] = 0.0
        model.R[2] = 0.0
        model.x = Var()
        model.CON1 = Constraint(expr=model.P[1] ** model.P[2] <= model.x)
        model.CON2 = Constraint(expr=model.Q[1] ** model.Q[2] <= model.x)
        model.CON3 = Constraint(expr=model.R[1] ** model.R[2] <= model.x)
        self.assertEqual(1.0, value(model.CON1[None].lower))
        self.assertEqual(1.0, value(model.CON2[None].lower))
        self.assertEqual(1.0, value(model.CON3[None].lower))
        model.P[1] = 3.0
        model.P[2] = 2.0
        model.Q[1] = 3.0
        model.Q[2] = 2.0
        model.R[1] = 3.0
        model.R[2] = 2.0
        self.assertEqual(9.0, value(model.CON1[None].lower))
        self.assertEqual(9.0, value(model.CON2[None].lower))
        self.assertEqual(9.0, value(model.CON3[None].lower))

    def test_mutable_abs_expr(self):
        model = ConcreteModel()
        model.P = Param([1, 2], initialize=-1.0, mutable=True)
        model.Q = Param([1, 2], default=-1.0, mutable=True)
        model.R = Param([1, 2], mutable=True)
        model.R[1] = -1.0
        model.R[2] = -1.0
        model.x = Var()
        model.CON1 = Constraint(expr=abs(model.P[1]) <= model.x)
        model.CON2 = Constraint(expr=abs(model.Q[1]) <= model.x)
        model.CON3 = Constraint(expr=abs(model.R[1]) <= model.x)
        self.assertEqual(1.0, value(model.CON1[None].lower))
        self.assertEqual(1.0, value(model.CON2[None].lower))
        self.assertEqual(1.0, value(model.CON3[None].lower))
        model.P[1] = -3.0
        model.Q[1] = -3.0
        model.R[1] = -3.0
        self.assertEqual(3.0, value(model.CON1[None].lower))
        self.assertEqual(3.0, value(model.CON2[None].lower))
        self.assertEqual(3.0, value(model.CON3[None].lower))

    def test_getting_value_may_insert(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertFalse(None in m.p)
        m.p.value = None
        self.assertTrue(None in m.p)
        m.q = Param()
        self.assertFalse(None in m.q)
        with self.assertRaises(ValueError):
            m.q.value
        self.assertFalse(None in m.q)
        m.qm = Param(mutable=True)
        self.assertFalse(None in m.qm)
        with self.assertRaises(ValueError):
            m.qm.value
        self.assertTrue(None in m.qm)
        m.r = Param([1], mutable=True)
        self.assertFalse(1 in m.r)
        m.r[1]
        self.assertTrue(1 in m.r)

    def test_using_None_in_params(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertEqual(len(m.p), 0)
        self.assertEqual(len(m.p._data), 0)
        m.p = None
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertIs(m.p.value, None)
        m.p = 1
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertEqual(m.p.value, 1)
        m = ConcreteModel()
        m.p = Param(mutable=True, initialize=None)
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertIs(m.p.value, None)
        m.p = 1
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertEqual(m.p.value, 1)
        m = ConcreteModel()
        m.p = Param(mutable=True, default=None)
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 0)
        self.assertIs(m.p.value, None)
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        m.p = 1
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertEqual(m.p.value, 1)
        m = ConcreteModel()
        m.p = Param(mutable=False, initialize=None)
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertIs(m.p.value, None)
        m = ConcreteModel()
        m.p = Param(mutable=False, default=None)
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 0)
        self.assertIs(m.p.value, None)
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 0)
        m = ConcreteModel()
        m.p = Param([1, 2], mutable=True)
        self.assertEqual(len(m.p), 0)
        self.assertEqual(len(m.p._data), 0)
        m.p[1] = None
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertIs(m.p[1].value, None)
        m.p[1] = 1
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertEqual(m.p[1].value, 1)
        m = ConcreteModel()
        m.p = Param([1, 2], mutable=True, initialize={1: None})
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertIs(m.p[1].value, None)
        m.p[2] = 1
        self.assertEqual(len(m.p), 2)
        self.assertEqual(len(m.p._data), 2)
        self.assertEqual(m.p[1].value, None)
        self.assertEqual(m.p[2].value, 1)
        m = ConcreteModel()
        m.p = Param([1, 2], mutable=True, default=None)
        self.assertEqual(len(m.p), 2)
        self.assertEqual(len(m.p._data), 0)
        self.assertIs(m.p[1].value, None)
        self.assertEqual(len(m.p), 2)
        self.assertEqual(len(m.p._data), 1)
        m.p[2] = 1
        self.assertEqual(len(m.p), 2)
        self.assertEqual(len(m.p._data), 2)
        self.assertIs(m.p[1].value, None)
        self.assertEqual(m.p[2].value, 1)
        m = ConcreteModel()
        m.p = Param([1, 2], mutable=False, initialize={1: None})
        self.assertEqual(len(m.p), 1)
        self.assertEqual(len(m.p._data), 1)
        self.assertIs(m.p[1], None)
        m = ConcreteModel()
        m.p = Param([1, 2], mutable=False, default=None)
        self.assertEqual(len(m.p), 2)
        self.assertEqual(len(m.p._data), 0)
        self.assertIs(m.p[1], None)
        self.assertEqual(len(m.p), 2)
        self.assertEqual(len(m.p._data), 0)