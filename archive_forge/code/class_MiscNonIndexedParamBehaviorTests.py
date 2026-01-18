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
class MiscNonIndexedParamBehaviorTests(unittest.TestCase):

    def test_mutable_self(self):
        model = ConcreteModel()
        model.Q = Param(initialize=0.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.Q <= model.x)
        self.assertEqual(0.0, value(model.CON[None].lower))
        model.Q = 1.0
        self.assertEqual(1.0, value(model.CON[None].lower))

    def test_mutable_display(self):
        model = ConcreteModel()
        model.Q = Param(initialize=0.0, mutable=True)
        self.assertEqual(model.Q.value, 0.0)
        f = StringIO()
        display(model.Q, f)
        tmp = f.getvalue().splitlines()
        val = float(tmp[-1].split(':')[-1].strip())
        self.assertEqual(model.Q.value, val)
        model.Q = 1.0
        self.assertEqual(model.Q.value, 1.0)
        f = StringIO()
        display(model.Q, f)
        tmp = f.getvalue().splitlines()
        val = float(tmp[-1].split(':')[-1].strip())
        self.assertEqual(model.Q.value, val)

    def test_mutable_pprint(self):
        model = ConcreteModel()
        model.Q = Param(initialize=0.0, mutable=True)
        self.assertEqual(model.Q.value, 0.0)
        buf = StringIO()
        model.Q.pprint(ostream=buf)
        val = float(buf.getvalue().splitlines()[-1].split(':')[-1].strip())
        self.assertEqual(model.Q.value, val)
        buf.buf = ''
        model.Q = 1.0
        self.assertEqual(model.Q.value, 1.0)
        model.Q.pprint(ostream=buf)
        val = float(buf.getvalue().splitlines()[-1].split(':')[-1].strip())
        self.assertEqual(model.Q.value, val)

    def test_mutable_sum_expr(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=0.0, mutable=True)
        model.Q2 = Param(initialize=0.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.Q1 + model.Q2 <= model.x)
        self.assertEqual(0.0, value(model.CON[None].lower))
        model.Q1 = 3.0
        model.Q2 = 2.0
        self.assertEqual(5.0, value(model.CON[None].lower))

    def test_mutable_prod_expr(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=0.0, mutable=True)
        model.Q2 = Param(initialize=0.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.Q1 * model.Q2 <= model.x)
        self.assertEqual(0.0, value(model.CON[None].lower))
        model.Q1 = 3.0
        model.Q2 = 2.0
        self.assertEqual(6.0, value(model.CON[None].lower))

    def test_mutable_pow_expr(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=1.0, mutable=True)
        model.Q2 = Param(initialize=1.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=model.Q1 ** model.Q2 <= model.x)
        self.assertEqual(1.0, value(model.CON[None].lower))
        model.Q1 = 3.0
        model.Q2 = 2.0
        self.assertEqual(9.0, value(model.CON[None].lower))

    def test_mutable_abs_expr(self):
        model = ConcreteModel()
        model.Q1 = Param(initialize=-1.0, mutable=True)
        model.x = Var()
        model.CON = Constraint(expr=abs(model.Q1) <= model.x)
        self.assertEqual(1.0, value(model.CON[None].lower))
        model.Q1 = -3.0
        self.assertEqual(3.0, value(model.CON[None].lower))