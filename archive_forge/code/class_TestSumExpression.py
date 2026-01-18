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
class TestSumExpression(unittest.TestCase):

    def setUp(self):
        self.m = ConcreteModel()
        self.m.I = RangeSet(5)
        self.m.a = Var(self.m.I, initialize=5)
        self.m.b = Var(self.m.I, initialize=10)
        self.m.p = Param(self.m.I, initialize=1, mutable=True)
        self.m.q = Param(self.m.I, initialize=3, mutable=False)

    def tearDown(self):
        self.m = None

    def test_deprecation(self):
        with LoggingIntercept() as LOG:
            e = quicksum((self.m.a[i] for i in self.m.a), linear=False)
        self.assertRegex(LOG.getvalue().replace('\n', ' '), 'DEPRECATED: The quicksum\\(linear=...\\) argument is deprecated and ignored.')
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, self.m.a[1])), MonomialTermExpression((1, self.m.a[2])), MonomialTermExpression((1, self.m.a[3])), MonomialTermExpression((1, self.m.a[4])), MonomialTermExpression((1, self.m.a[5]))]))

    def test_summation1(self):
        e = quicksum((self.m.a[i] for i in self.m.a))
        self.assertEqual(e(), 25)
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, self.m.a[1])), MonomialTermExpression((1, self.m.a[2])), MonomialTermExpression((1, self.m.a[3])), MonomialTermExpression((1, self.m.a[4])), MonomialTermExpression((1, self.m.a[5]))]))

    def test_summation2(self):
        e = quicksum((self.m.p[i] * self.m.a[i] for i in self.m.a))
        self.assertEqual(e(), 25)
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((self.m.p[1], self.m.a[1])), MonomialTermExpression((self.m.p[2], self.m.a[2])), MonomialTermExpression((self.m.p[3], self.m.a[3])), MonomialTermExpression((self.m.p[4], self.m.a[4])), MonomialTermExpression((self.m.p[5], self.m.a[5]))]))

    def test_summation3(self):
        e = quicksum((self.m.q[i] * self.m.a[i] for i in self.m.a))
        self.assertEqual(e(), 75)
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((3, self.m.a[1])), MonomialTermExpression((3, self.m.a[2])), MonomialTermExpression((3, self.m.a[3])), MonomialTermExpression((3, self.m.a[4])), MonomialTermExpression((3, self.m.a[5]))]))

    def test_summation4(self):
        e = quicksum((self.m.a[i] * self.m.b[i] for i in self.m.a))
        self.assertEqual(e(), 250)
        self.assertExpressionsEqual(e, SumExpression([ProductExpression((self.m.a[1], self.m.b[1])), ProductExpression((self.m.a[2], self.m.b[2])), ProductExpression((self.m.a[3], self.m.b[3])), ProductExpression((self.m.a[4], self.m.b[4])), ProductExpression((self.m.a[5], self.m.b[5]))]))

    def test_summation5(self):
        e = quicksum((self.m.b[i] / self.m.a[i] for i in self.m.a))
        self.assertEqual(e(), 10)
        self.assertExpressionsEqual(e, SumExpression([DivisionExpression((self.m.b[1], self.m.a[1])), DivisionExpression((self.m.b[2], self.m.a[2])), DivisionExpression((self.m.b[3], self.m.a[3])), DivisionExpression((self.m.b[4], self.m.a[4])), DivisionExpression((self.m.b[5], self.m.a[5]))]))

    def test_summation6(self):
        e = quicksum((self.m.a[i] / self.m.p[i] for i in self.m.a))
        self.assertEqual(e(), 25)
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((NPV_DivisionExpression((1, self.m.p[1])), self.m.a[1])), MonomialTermExpression((NPV_DivisionExpression((1, self.m.p[2])), self.m.a[2])), MonomialTermExpression((NPV_DivisionExpression((1, self.m.p[3])), self.m.a[3])), MonomialTermExpression((NPV_DivisionExpression((1, self.m.p[4])), self.m.a[4])), MonomialTermExpression((NPV_DivisionExpression((1, self.m.p[5])), self.m.a[5]))]))

    def test_summation7(self):
        e = quicksum((self.m.p[i] * self.m.q[i] for i in self.m.I), linear=False)
        self.assertEqual(e(), 15)
        self.assertExpressionsEqual(e, NPV_SumExpression([NPV_ProductExpression((self.m.p[1], 3)), NPV_ProductExpression((self.m.p[2], 3)), NPV_ProductExpression((self.m.p[3], 3)), NPV_ProductExpression((self.m.p[4], 3)), NPV_ProductExpression((self.m.p[5], 3))]))

    def test_quicksum_reject_noniterable(self):
        with LoggingIntercept() as LOG:
            with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
                quicksum(1)
        self.assertEqual(LOG.getvalue(), 'The argument `args` to quicksum() is not iterable!\n')

    def test_quicksum_exception_exposure(self):
        ex0 = Exception()

        def f():
            raise ex0
        with self.assertRaises(Exception) as cm:
            quicksum((f() for i in [1, 2, 3]), linear=None)
        self.assertIs(cm.exception, ex0)
        with self.assertRaises(Exception) as cm:
            quicksum((f() for i in [1, 2, 3]), linear=True)
        self.assertIs(cm.exception, ex0)
        with self.assertRaises(Exception) as cm:
            quicksum((f() for i in [1, 2, 3]), linear=False)
        self.assertIs(cm.exception, ex0)
        with self.assertRaises(Exception) as cm:
            quicksum((f() for i in [1, 2, 3]), start=self.m.a[1])
        self.assertIs(cm.exception, ex0)