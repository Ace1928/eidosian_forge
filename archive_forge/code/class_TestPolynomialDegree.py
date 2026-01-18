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
class TestPolynomialDegree(unittest.TestCase):

    def setUp(self):

        def d_fn(model):
            return model.c + model.c
        self.model = ConcreteModel()
        self.model.a = Var(initialize=1.0)
        self.model.b = Var(initialize=2.0)
        self.model.c = Param(initialize=3.0, mutable=True)
        self.model.d = Param(initialize=d_fn, mutable=True)
        self.model.e = Param(mutable=True)
        self.instance = self.model

    def tearDown(self):
        self.model = None
        self.instance = None

    def test_param(self):
        self.assertEqual(self.model.d.polynomial_degree(), 0)

    def test_var(self):
        self.model.a.fixed = False
        self.assertEqual(self.model.a.polynomial_degree(), 1)
        self.model.a.fixed = True
        self.assertEqual(self.model.a.polynomial_degree(), 0)

    def test_simple_sum(self):
        expr = self.model.c + self.model.d
        self.assertEqual(expr.polynomial_degree(), 0)
        expr = self.model.a + self.model.b
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = False
        expr = self.model.a + self.model.c
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)

    def test_linearsum(self):
        m = ConcreteModel()
        A = range(5)
        m.v = Var(A)
        e = quicksum((m.v[i] for i in A))
        self.assertIs(e.__class__, LinearExpression)
        self.assertEqual(e.polynomial_degree(), 1)
        e = quicksum((i * m.v[i] for i in A))
        self.assertIs(e.__class__, LinearExpression)
        self.assertEqual(e.polynomial_degree(), 1)
        e = quicksum((1 for i in A))
        self.assertIs(e.__class__, int)
        self.assertEqual(polynomial_degree(e), 0)

    def test_relational_ops(self):
        expr = self.model.c < self.model.d
        self.assertEqual(expr.polynomial_degree(), 0)
        expr = self.model.a <= self.model.d
        self.assertEqual(expr.polynomial_degree(), 1)
        expr = self.model.a * self.model.a >= self.model.b
        self.assertEqual(expr.polynomial_degree(), 2)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = False
        expr = self.model.a > self.model.a * self.model.b
        self.assertEqual(expr.polynomial_degree(), 2)
        self.model.b.fixed = True
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.b.fixed = False
        expr = self.model.a == self.model.a * self.model.b
        self.assertEqual(expr.polynomial_degree(), 2)
        self.model.b.fixed = True
        self.assertEqual(expr.polynomial_degree(), 1)

    def test_simple_product(self):
        expr = self.model.c * self.model.d
        self.assertEqual(expr.polynomial_degree(), 0)
        expr = self.model.a * self.model.b
        self.assertEqual(expr.polynomial_degree(), 2)
        expr = self.model.a * self.model.c
        self.assertEqual(expr.polynomial_degree(), 1)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        self.model.a.fixed = False
        expr = self.model.a / self.model.c
        self.assertEqual(expr.polynomial_degree(), 1)
        expr = self.model.c / self.model.a
        self.assertEqual(expr.polynomial_degree(), None)
        self.model.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        self.model.a.fixed = False

    def test_nested_expr(self):
        expr1 = self.model.c * self.model.d
        expr2 = expr1 + expr1
        self.assertEqual(expr2.polynomial_degree(), 0)
        expr1 = self.model.a * self.model.b
        expr2 = expr1 + expr1
        self.assertEqual(expr2.polynomial_degree(), 2)
        self.model.a.fixed = True
        self.assertEqual(expr2.polynomial_degree(), 1)
        self.model.a.fixed = False
        expr1 = self.model.c + self.model.d
        expr2 = expr1 * expr1
        self.assertEqual(expr2.polynomial_degree(), 0)
        expr1 = self.model.a + self.model.b
        expr2 = expr1 * expr1
        self.assertEqual(expr2.polynomial_degree(), 2)
        self.model.a.fixed = True
        self.assertEqual(expr2.polynomial_degree(), 2)
        self.model.b.fixed = True
        self.assertEqual(expr2.polynomial_degree(), 0)

    def test_misc_operators(self):
        expr = -(self.model.a * self.model.b)
        self.assertEqual(expr.polynomial_degree(), 2)

    def test_nonpolynomial_abs(self):
        expr1 = abs(self.model.a * self.model.b)
        self.assertEqual(expr1.polynomial_degree(), None)
        expr2 = self.model.a + self.model.b * abs(self.model.b)
        self.assertEqual(expr2.polynomial_degree(), None)
        expr3 = self.model.a * (self.model.b + abs(self.model.b))
        self.assertEqual(expr3.polynomial_degree(), None)
        self.model.a.fixed = True
        self.assertEqual(expr1.polynomial_degree(), None)
        self.assertEqual(expr2.polynomial_degree(), None)
        self.assertEqual(expr3.polynomial_degree(), None)
        self.model.b.fixed = True
        self.assertEqual(expr1.polynomial_degree(), 0)
        self.assertEqual(expr2.polynomial_degree(), 0)
        self.assertEqual(expr3.polynomial_degree(), 0)
        self.model.a.fixed = False
        self.assertEqual(expr1.polynomial_degree(), None)
        self.assertEqual(expr2.polynomial_degree(), 1)
        self.assertEqual(expr3.polynomial_degree(), 1)

    def test_nonpolynomial_pow(self):
        m = self.instance
        expr = pow(m.a, m.b)
        self.assertEqual(expr.polynomial_degree(), None)
        m.b.fixed = True
        self.assertEqual(expr.polynomial_degree(), 2)
        m.b.value = 0
        self.assertEqual(expr.polynomial_degree(), 0)
        m.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        m.b.fixed = False
        self.assertEqual(expr.polynomial_degree(), None)
        m.a.fixed = False
        expr = pow(m.a, 1)
        self.assertEqual(expr.polynomial_degree(), 1)
        expr = pow(m.a, 2)
        self.assertEqual(expr.polynomial_degree(), 2)
        expr = pow(m.a * m.a, 2)
        self.assertEqual(expr.polynomial_degree(), 4)
        expr = pow(m.a * m.a, 2.1)
        self.assertEqual(expr.polynomial_degree(), None)
        expr = pow(m.a * m.a, -1)
        self.assertEqual(expr.polynomial_degree(), None)
        expr = pow(2 ** m.a, 1)
        self.assertEqual(expr.polynomial_degree(), None)
        expr = pow(2 ** m.a, 0)
        self.assertEqual(expr.polynomial_degree(), 0)
        expr = pow(m.a, m.e)
        self.assertEqual(expr.polynomial_degree(), None)

    def test_Expr_if(self):
        m = self.instance
        expr = Expr_if(1, m.a ** 3, m.a ** 2)
        self.assertEqual(expr.polynomial_degree(), 3)
        m.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        m.a.fixed = False
        expr = Expr_if(0, m.a ** 3, m.a ** 2)
        self.assertEqual(expr.polynomial_degree(), 2)
        m.a.fixed = True
        self.assertEqual(expr.polynomial_degree(), 0)
        m.a.fixed = False
        expr = Expr_if(m.a, m.b, m.b ** 2)
        self.assertEqual(expr.polynomial_degree(), None)
        m.a.fixed = True
        m.a.value = 1
        self.assertEqual(expr.polynomial_degree(), 1)
        expr = Expr_if(m.e, 1, 0)
        self.assertEqual(expr.polynomial_degree(), 0)
        expr = Expr_if(m.e, m.a, 0)
        self.assertEqual(expr.polynomial_degree(), 0)
        expr = Expr_if(m.e, 5 * m.b, 1 + m.b)
        self.assertEqual(expr.polynomial_degree(), 1)
        expr = Expr_if(m.e, m.b, 0)
        self.assertEqual(expr.polynomial_degree(), None)