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
class TestPrettyPrinter_oldStyle(unittest.TestCase):
    _save = None

    def setUp(self):
        TestPrettyPrinter_oldStyle._save = expr_common.TO_STRING_VERBOSE
        expr_common.TO_STRING_VERBOSE = True

    def tearDown(self):
        expr_common.TO_STRING_VERBOSE = TestPrettyPrinter_oldStyle._save

    def test_sum(self):
        model = ConcreteModel()
        model.a = Var()
        model.p = Param(mutable=True)
        expr = 5 + model.a + model.a
        self.assertEqual('sum(5, mon(1, a), mon(1, a))', str(expr))
        expr += 5
        self.assertEqual('sum(5, mon(1, a), mon(1, a), 5)', str(expr))
        expr = 2 + model.p
        self.assertEqual('sum(2, p)', str(expr))

    def test_linearsum(self):
        model = ConcreteModel()
        A = range(5)
        model.a = Var(A)
        model.p = Param(A, initialize=2, mutable=True)
        expr = quicksum((i * model.a[i] for i in A))
        self.assertEqual('sum(mon(0, a[0]), mon(1, a[1]), mon(2, a[2]), mon(3, a[3]), mon(4, a[4]))', str(expr))
        expr = quicksum(((i - 2) * model.a[i] for i in A))
        self.assertEqual('sum(mon(-2, a[0]), mon(-1, a[1]), mon(0, a[2]), mon(1, a[3]), mon(2, a[4]))', str(expr))
        expr = quicksum((model.a[i] for i in A))
        self.assertEqual('sum(mon(1, a[0]), mon(1, a[1]), mon(1, a[2]), mon(1, a[3]), mon(1, a[4]))', str(expr))
        model.p[1].value = 0
        model.p[3].value = 3
        expr = quicksum((model.p[i] * model.a[i] if i != 3 else model.p[i] for i in A))
        self.assertEqual('sum(mon(2, a[0]), mon(0, a[1]), mon(2, a[2]), 3, mon(2, a[4]))', expression_to_string(expr, compute_values=True))
        self.assertEqual('sum(mon(p[0], a[0]), mon(p[1], a[1]), mon(p[2], a[2]), p[3], mon(p[4], a[4]))', expression_to_string(expr, compute_values=False))

    def test_expr(self):
        model = ConcreteModel()
        model.a = Var()
        expr = 5 * model.a * model.a
        self.assertEqual('prod(mon(5, a), a)', str(expr))
        expr = 5 * model.a / model.a
        self.assertEqual('div(mon(5, a), a)', str(expr))
        expr = expr / model.a
        self.assertEqual('div(div(mon(5, a), a), a)', str(expr))
        expr = 5 * model.a / model.a / 2
        self.assertEqual('div(div(mon(5, a), a), 2)', str(expr))

    def test_other(self):
        model = ConcreteModel()
        model.a = Var()
        model.x = ExternalFunction(library='foo.so', function='bar')
        expr = model.x(model.a, 1, 'foo', [])
        self.assertEqual("x(a, 1, 'foo', '[]')", str(expr))

    def test_inequality(self):
        model = ConcreteModel()
        model.a = Var()
        expr = 5 < model.a
        self.assertEqual('5  <  a', str(expr))
        expr = model.a >= 5
        self.assertEqual('5  <=  a', str(expr))
        expr = expr < 10
        self.assertEqual('5  <=  a  <  10', str(expr))
        expr = 5 <= model.a + 5
        self.assertEqual('5  <=  sum(mon(1, a), 5)', str(expr))
        expr = expr < 10
        self.assertEqual('5  <=  sum(mon(1, a), 5)  <  10', str(expr))

    def test_equality(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Param(initialize=5, mutable=True)
        expr = model.a == model.b
        self.assertEqual('a  ==  b', str(expr))
        expr = model.b == model.a
        self.assertEqual('b  ==  a', str(expr))
        expr = 5 == model.a
        self.assertEqual('a  ==  5', str(expr))
        expr = model.a == 10
        self.assertEqual('a  ==  10', str(expr))
        expr = 5 == model.a + 5
        self.assertEqual('sum(mon(1, a), 5)  ==  5', str(expr))
        expr = model.a + 5 == 5
        self.assertEqual('sum(mon(1, a), 5)  ==  5', str(expr))

    def test_getitem(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 9)
        m.x = Var(m.I, initialize=lambda m, i: i + 1)
        m.P = Param(m.I, initialize=lambda m, i: 10 - i, mutable=True)
        t = IndexTemplate(m.I)
        e = m.x[t + m.P[t + 1]] + 3
        self.assertEqual('sum(getitem(x, sum({I}, getitem(P, sum({I}, 1)))), 3)', str(e))

    def test_small_expression(self):
        model = AbstractModel()
        model.a = Var()
        model.b = Param(initialize=2, mutable=True)
        instance = model.create_instance()
        expr = instance.a + 1
        expr = expr - 1
        expr = expr * instance.a
        expr = expr / instance.a
        expr = expr ** instance.b
        expr = 1 - expr
        expr = 1 + expr
        expr = 2 * expr
        expr = 2 / expr
        expr = 2 ** expr
        expr = -expr
        expr = +expr
        expr = abs(expr)
        self.assertEqual('abs(neg(pow(2, div(2, prod(2, sum(1, neg(pow(div(prod(sum(mon(1, a), 1, -1), a), a), b)), 1))))))', str(expr))