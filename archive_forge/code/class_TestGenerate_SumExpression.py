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
class TestGenerate_SumExpression(unittest.TestCase):

    def test_simpleSum(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a + m.b
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b))]))
        self.assertRaises(KeyError, e.arg, 3)

    def test_simpleSum_API(self):
        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        e = m.a + m.b
        e += 2 * m.a
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b)), MonomialTermExpression((2, m.a))]))

    def test_constSum(self):
        m = AbstractModel()
        m.a = Var()
        self.assertExpressionsEqual(m.a + 5, LinearExpression([MonomialTermExpression((1, m.a)), 5]))
        self.assertExpressionsEqual(5 + m.a, LinearExpression([5, MonomialTermExpression((1, m.a))]))

    def test_nestedSum(self):
        expectedType = SumExpression
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()
        e1 = m.a + m.b
        e = e1 + 5
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b)), 5]))
        e1 = m.a + m.b
        e = 5 + e1
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b)), 5]))
        e1 = m.a + m.b
        e = e1 + m.c
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b)), MonomialTermExpression((1, m.c))]))
        e1 = m.a + m.b
        e = m.c + e1
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b)), MonomialTermExpression((1, m.c))]))
        e1 = m.a + m.b
        e2 = m.c + m.d
        e = e1 + e2
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b)), MonomialTermExpression((1, m.c)), MonomialTermExpression((1, m.d))]))

    def test_nestedSum2(self):
        expectedType = SumExpression
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()
        e1 = m.a + m.b
        e = 2 * e1 + m.c
        self.assertExpressionsEqual(e, SumExpression([ProductExpression((2, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b))]))), m.c]))
        e1 = m.a + m.b
        e = 3 * (2 * e1 + m.c)
        self.assertExpressionsEqual(e, ProductExpression((3, SumExpression([ProductExpression((2, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b))]))), m.c]))))

    def test_trivialSum(self):
        m = AbstractModel()
        m.a = Var()
        e = m.a + 0
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)
        e = 0 + m.a
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)
        e = m.a + m.a
        f = e + 0
        self.assertEqual(id(e), id(f))

    def test_sumOf_nestedTrivialProduct(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        e1 = m.a * 5
        e = e1 + m.b
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((5, m.a)), MonomialTermExpression((1, m.b))]))
        e = m.b + e1
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.b)), MonomialTermExpression((5, m.a))]))
        e2 = m.b + m.c
        e = e1 + e2
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.b)), MonomialTermExpression((1, m.c)), MonomialTermExpression((5, m.a))]))
        e2 = m.b + m.c
        e = e2 + e1
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.b)), MonomialTermExpression((1, m.c)), MonomialTermExpression((5, m.a))]))

    def test_simpleDiff(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a - m.b
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((-1, m.b))]))

    def test_constDiff(self):
        m = AbstractModel()
        m.a = Var()
        self.assertExpressionsEqual(m.a - 5, LinearExpression([MonomialTermExpression((1, m.a)), -5]))
        self.assertExpressionsEqual(5 - m.a, LinearExpression([5, MonomialTermExpression((-1, m.a))]))

    def test_paramDiff(self):
        m = AbstractModel()
        m.a = Var()
        m.p = Param()
        e = m.a - m.p
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), NPV_NegationExpression((m.p,))]))
        e = m.p - m.a
        self.assertExpressionsEqual(e, LinearExpression([m.p, MonomialTermExpression((-1, m.a))]))

    def test_constparamDiff(self):
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(initialize=0)
        e = m.a - m.p
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)
        e = m.p - m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), -1)
        self.assertIs(e.arg(1), m.a)

    def test_termDiff(self):
        m = ConcreteModel()
        m.a = Var()
        e = 5 - 2 * m.a
        self.assertExpressionsEqual(e, LinearExpression([5, MonomialTermExpression((-2, m.a))]))

    def test_nestedDiff(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()
        e1 = m.a - m.b
        e = e1 - 5
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((-1, m.b)), -5]))
        e1 = m.a - m.b
        e = 5 - e1
        self.assertExpressionsEqual(e, SumExpression([5, NegationExpression((LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((-1, m.b))]),))]))
        e1 = m.a - m.b
        e = e1 - m.c
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((-1, m.b)), MonomialTermExpression((-1, m.c))]))
        e1 = m.a - m.b
        e = m.c - e1
        self.assertExpressionsEqual(e, SumExpression([m.c, NegationExpression((LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((-1, m.b))]),))]))
        e1 = m.a - m.b
        e2 = m.c - m.d
        e = e1 - e2
        self.assertExpressionsEqual(e, SumExpression([LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((-1, m.b))]), NegationExpression((LinearExpression([MonomialTermExpression((1, m.c)), MonomialTermExpression((-1, m.d))]),))]))

    def test_negation_param(self):
        m = AbstractModel()
        m.p = Param()
        e = -m.p
        self.assertIs(type(e), NPV_NegationExpression)
        e = -e
        self.assertTrue(isinstance(e, Param))

    def test_negation_mutableparam(self):
        m = AbstractModel()
        m.p = Param(mutable=True, initialize=1.0)
        e = -m.p
        self.assertExpressionsEqual(e, NPV_NegationExpression((m.p,)))
        self.assertExpressionsEqual(-e, m.p)

    def test_negation_terms(self):
        m = AbstractModel()
        m.v = Var()
        m.p = Param(mutable=True, initialize=1.0)
        e = -m.p * m.v
        self.assertExpressionsEqual(e, MonomialTermExpression((NPV_NegationExpression((m.p,)), m.v)))
        self.assertExpressionsEqual(-e, MonomialTermExpression((m.p, m.v)))
        e = -5 * m.v
        self.assertExpressionsEqual(e, MonomialTermExpression((-5, m.v)))
        self.assertExpressionsEqual(-e, MonomialTermExpression((5, m.v)))

    def test_trivialDiff(self):
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(mutable=True)
        e = m.a - 0
        self.assertIs(type(e), type(m.a))
        self.assertIs(e, m.a)
        e = 0 - m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), -1)
        self.assertIs(e.arg(1), m.a)
        e = m.p - 0
        self.assertIs(type(e), type(m.p))
        self.assertIs(e, m.p)
        e = 0 - m.p
        self.assertIs(type(e), NPV_NegationExpression)
        self.assertEqual(e.nargs(), 1)
        self.assertIs(e.arg(0), m.p)
        e = 0 - 5 * m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), -5)
        e = 0 - m.p * m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), NPV_NegationExpression)
        self.assertIs(e.arg(0).arg(0), m.p)
        e = 0 - m.a * m.a
        self.assertIs(type(e), NegationExpression)
        self.assertEqual(e.nargs(), 1)
        self.assertIs(type(e.arg(0)), ProductExpression)

    def test_sumOf_nestedTrivialProduct2(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        e1 = m.a * 5
        e = e1 - m.b
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), e1)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 7)
        e1 = m.a * 5
        e = m.b - e1
        self.assertIs(type(e), SumExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(e.arg(1).arg(0), -5)
        self.assertIs(e.arg(1).arg(1), m.a)
        self.assertEqual(e.size(), 5)
        e1 = m.a * 5
        e2 = m.b - m.c
        e = e1 - e2
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), e1)
        self.assertIs(type(e.arg(1)), NegationExpression)
        self.assertIs(e.arg(1).arg(0), e2)
        self.assertEqual(e.size(), 10)
        e1 = m.a * 5
        e2 = m.b - m.c
        e = e2 - e1
        self.assertIs(type(e), SumExpression)
        self.assertIs(e.arg(0), m.b)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertEqual(e.arg(1).arg(0), -1)
        self.assertIs(e.arg(1).arg(1), m.c)
        self.assertIs(e.arg(2).arg(0), -5)
        self.assertIs(e.arg(2).arg(1), m.a)
        self.assertEqual(e.size(), 8)

    def test_sumOf_nestedTrivialProduct2(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.p = Param(initialize=5, mutable=True)
        e1 = m.a * m.p
        e = e1 - m.b
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((m.p, m.a)), MonomialTermExpression((-1, m.b))]))
        e1 = m.a * m.p
        e = m.b - e1
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.b)), MonomialTermExpression((NPV_NegationExpression((m.p,)), m.a))]))
        e1 = m.a * m.p
        e2 = m.b - m.c
        e = e1 - e2
        self.assertExpressionsEqual(e, SumExpression([MonomialTermExpression((m.p, m.a)), NegationExpression((LinearExpression([MonomialTermExpression((1, m.b)), MonomialTermExpression((-1, m.c))]),))]))
        e1 = m.a * m.p
        e2 = m.b - m.c
        e = e2 - e1
        self.maxDiff = None
        self.assertExpressionsEqual(e, LinearExpression([MonomialTermExpression((1, m.b)), MonomialTermExpression((-1, m.c)), MonomialTermExpression((NPV_NegationExpression((m.p,)), m.a))]))