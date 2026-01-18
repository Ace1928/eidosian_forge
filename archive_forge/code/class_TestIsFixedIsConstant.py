import copy
import pickle
import math
import os
from collections import defaultdict
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.environ import (
from pyomo.kernel import variable, expression, objective
from pyomo.core.expr.expr_common import ExpressionType, clone_counter
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.core.expr.relational_expr import RelationalExpression, EqualityExpression
from pyomo.common.errors import PyomoException
from pyomo.core.expr.visitor import expression_to_string, clone_expression
from pyomo.core.expr import Expr_if
from pyomo.core.base.label import NumericLabeler
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr import expr_common
from pyomo.core.base.var import _GeneralVarData
from pyomo.repn import generate_standard_repn
from pyomo.core.expr.numvalue import NumericValue
class TestIsFixedIsConstant(unittest.TestCase):

    def setUp(self):

        def d_fn(model):
            return model.c + model.c
        self.model = AbstractModel()
        self.model.a = Var(initialize=1.0)
        self.model.b = Var(initialize=2.0)
        self.model.c = Param(initialize=1, mutable=True)
        self.model.d = Param(initialize=d_fn, mutable=True)
        self.model.e = Param(initialize=d_fn, mutable=False)
        self.model.f = Param(initialize=0, mutable=True)
        self.model.g = Var(initialize=0)
        self.instance = self.model.create_instance()

    def tearDown(self):
        self.model = None
        self.instance = None

    def test_simple_sum(self):
        expr = self.instance.c + self.instance.d
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        expr = self.instance.e + self.instance.d
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        expr = self.instance.e + self.instance.e
        self.assertEqual(is_fixed(expr), True)
        self.assertEqual(is_constant(expr), True)
        self.assertEqual(is_potentially_variable(expr), False)
        expr = self.instance.a + self.instance.b
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        self.instance.b.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

    def test_linear_sum(self):
        m = ConcreteModel()
        A = range(5)
        m.v = Var(A)
        e = quicksum((m.v[i] for i in A))
        self.assertEqual(e.is_fixed(), False)
        for i in A:
            m.v[i].fixed = True
        self.assertEqual(e.is_fixed(), True)
        with linear_expression() as e:
            e += 1
        self.assertIs(e.__class__, NPV_SumExpression)
        self.assertEqual(e.is_fixed(), True)

    def test_simple_product(self):
        expr = self.instance.c * self.instance.d
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        expr = self.instance.a * self.instance.c
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = self.instance.f * self.instance.b
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = self.instance.a * self.instance.b
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        self.instance.b.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.instance.a.fixed = False
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = self.instance.a * self.instance.g
        self.instance.a.fixed = False
        self.instance.g.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = self.instance.a / self.instance.c
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        self.instance.a.fixed = False
        expr = self.instance.c / self.instance.a
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        self.instance.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

    def test_misc_operators(self):
        expr = -(self.instance.a * self.instance.b)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

    def test_polynomial_external_func(self):
        model = ConcreteModel()
        model.a = Var()
        model.p = Param(initialize=1, mutable=True)
        model.x = ExternalFunction(library='foo.so', function='bar')
        expr = model.x(2 * model.a, 1, 'foo', [])
        self.assertEqual(expr.polynomial_degree(), None)
        expr = model.x(2 * model.p, 1, 'foo', [])
        self.assertEqual(expr.polynomial_degree(), 0)

    def test_getitem(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 9)
        m.x = Var(m.I, initialize=lambda m, i: i + 1)
        m.P = Param(m.I, initialize=lambda m, i: 10 - i, mutable=True)
        t = IndexTemplate(m.I)
        e = m.x[t]
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(e.is_fixed(), False)
        e = m.x[t + m.P[t + 1]] + 3
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(e.is_fixed(), False)
        for i in m.I:
            m.x[i].fixed = True
        e = m.x[t]
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(e.is_fixed(), True)
        e = m.x[t + m.P[t + 1]] + 3
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(e.is_fixed(), True)
        e = m.P[t + 1] + 3
        self.assertEqual(e.is_constant(), False)
        self.assertEqual(m.P[t + 1].is_potentially_variable(), False)
        self.assertEqual(e.is_potentially_variable(), False)
        self.assertEqual(e.is_fixed(), True)

    def test_nonpolynomial_abs(self):
        expr1 = abs(self.instance.a * self.instance.b)
        self.assertEqual(expr1.is_fixed(), False)
        self.assertEqual(expr1.is_constant(), False)
        self.assertEqual(expr1.is_potentially_variable(), True)
        expr2 = self.instance.a + self.instance.b * abs(self.instance.b)
        self.assertEqual(expr2.is_fixed(), False)
        self.assertEqual(expr2.is_constant(), False)
        self.assertEqual(expr2.is_potentially_variable(), True)
        expr3 = self.instance.a * (self.instance.b + abs(self.instance.b))
        self.assertEqual(expr3.is_fixed(), False)
        self.assertEqual(expr3.is_constant(), False)
        self.assertEqual(expr3.is_potentially_variable(), True)
        self.instance.a.fixed = True
        self.assertEqual(expr1.is_fixed(), False)
        self.assertEqual(expr1.is_constant(), False)
        self.assertEqual(expr1.is_potentially_variable(), True)
        self.assertEqual(expr2.is_fixed(), False)
        self.assertEqual(expr2.is_constant(), False)
        self.assertEqual(expr2.is_potentially_variable(), True)
        self.assertEqual(expr3.is_fixed(), False)
        self.assertEqual(expr3.is_constant(), False)
        self.assertEqual(expr3.is_potentially_variable(), True)
        self.instance.b.fixed = True
        self.assertEqual(expr1.is_fixed(), True)
        self.assertEqual(expr1.is_constant(), False)
        self.assertEqual(expr1.is_potentially_variable(), True)
        self.assertEqual(expr2.is_fixed(), True)
        self.assertEqual(expr2.is_constant(), False)
        self.assertEqual(expr2.is_potentially_variable(), True)
        self.assertEqual(expr3.is_fixed(), True)
        self.assertEqual(expr3.is_constant(), False)
        self.assertEqual(expr3.is_potentially_variable(), True)
        self.instance.a.fixed = False
        self.assertEqual(expr1.is_fixed(), False)
        self.assertEqual(expr1.is_constant(), False)
        self.assertEqual(expr1.is_potentially_variable(), True)
        self.assertEqual(expr2.is_fixed(), False)
        self.assertEqual(expr2.is_constant(), False)
        self.assertEqual(expr2.is_potentially_variable(), True)
        self.assertEqual(expr3.is_fixed(), False)
        self.assertEqual(expr3.is_constant(), False)
        self.assertEqual(expr3.is_potentially_variable(), True)

    def test_nonpolynomial_pow(self):
        m = self.instance
        expr = pow(m.d, m.e)
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), False)
        expr = pow(m.a, m.b)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.b.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.b.fixed = False
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = False
        expr = pow(m.a, 1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = pow(m.a, 2)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = pow(m.a * m.a, 2)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = pow(m.a * m.a, 2.1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = pow(m.a * m.a, -1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = pow(2 ** m.a, 1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr = pow(2 ** m.a, 0)
        self.assertEqual(is_fixed(expr), True)
        self.assertEqual(is_constant(expr), False)
        self.assertEqual(is_potentially_variable(expr), True)

    def test_Expr_if(self):
        m = self.instance
        expr = Expr_if(1, m.a, m.e)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = False
        expr = Expr_if(0, m.a, m.e)
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), True)
        m.a.fixed = True
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), True)
        m.a.fixed = False
        expr = Expr_if(m.a, m.b, m.b)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = True
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fixed = False

    def test_LinearExpr(self):
        m = self.instance
        expr = m.a + m.b
        self.assertIs(type(expr), LinearExpression)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.fix(1)
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.b.fix(1)
        self.assertEqual(expr.is_fixed(), True)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.unfix()
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        m.a.unfix()
        expr -= m.a
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)
        expr -= m.b
        self.assertEqual(expr.is_fixed(), False)
        self.assertEqual(expr.is_constant(), False)
        self.assertEqual(expr.is_potentially_variable(), True)

    def test_expression(self):
        m = ConcreteModel()
        m.x = Expression()
        e = m.x
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e = m.x + 1
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e = m.x ** 2
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)
        e = m.x ** 2 / (m.x + 1)
        self.assertEqual(e.is_potentially_variable(), True)
        self.assertEqual(is_potentially_variable(e), True)

    def test_external_func(self):
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.p = Param(initialize=1, mutable=True)
        m.x = ExternalFunction(library='foo.so', function='bar')
        e = m.x(m.a, 1, 'foo bar', [])
        self.assertEqual(e.is_potentially_variable(), True)
        e = m.x(m.p, 1, 'foo bar', [])
        self.assertEqual(e.is_potentially_variable(), False)