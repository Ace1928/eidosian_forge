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
class TestGenerate_ProductExpression(unittest.TestCase):

    def test_simpleProduct(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a * m.b
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)

    def test_constProduct(self):
        m = AbstractModel()
        m.a = Var()
        e = m.a * 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)
        e = 5 * m.a
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), 5)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)

    def test_nestedProduct(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()
        e1 = m.a * m.b
        e = e1 * 5
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(1), 5)
        self.assertIs(type(e.arg(0)), ProductExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a * m.b
        e = 5 * e1
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a * m.b
        e = e1 * m.c
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(1), m.c)
        self.assertIs(type(e.arg(0)), ProductExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a * m.b
        e = m.c * e1
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.c)
        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a * m.b
        e2 = m.c * m.d
        e = e1 * e2
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), ProductExpression)
        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertIs(e.arg(1).arg(0), m.c)
        self.assertIs(e.arg(1).arg(1), m.d)
        self.assertEqual(e.size(), 7)

    def test_nestedProduct2(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()
        e1 = m.a + m.b
        e2 = m.c + e1
        e3 = e1 + m.d
        e = e2 * e3
        self.assertExpressionsEqual(e, ProductExpression((LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b)), MonomialTermExpression((1, m.c))]), LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b)), MonomialTermExpression((1, m.d))]))))
        self.assertIs(e1._args_, e2._args_)
        self.assertIsNot(e1._args_, e3._args_)
        self.assertIs(e1._args_, e.arg(0)._args_)
        self.assertIs(e.arg(0).arg(0), e.arg(1).arg(0))
        self.assertIs(e.arg(0).arg(1), e.arg(1).arg(1))
        e1 = m.a + m.b
        e2 = m.c * e1
        e3 = e1 * m.d
        e = e2 * e3
        inner = LinearExpression([MonomialTermExpression((1, m.a)), MonomialTermExpression((1, m.b))])
        self.assertExpressionsEqual(e, ProductExpression((ProductExpression((m.c, inner)), ProductExpression((inner, m.d)))))
        self.assertIs(e.arg(0).arg(1), e.arg(1).arg(0))

    def test_nestedProduct3(self):
        m = AbstractModel()
        m.a = Param(mutable=True)
        m.b = Var()
        m.c = Var()
        m.d = Var()
        e1 = 3 * m.b
        e = e1 * 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 15)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)
        e1 = m.a * m.b
        e = e1 * 5
        self.assertExpressionsEqual(e, MonomialTermExpression((NPV_ProductExpression((m.a, 5)), m.b)))
        e1 = 3 * m.b
        e = 5 * e1
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 15)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)
        e1 = m.a * m.b
        e = 5 * e1
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), NPV_ProductExpression)
        self.assertEqual(e.arg(0).arg(0), 5)
        self.assertIs(e.arg(0).arg(1), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a * m.b
        e = e1 * m.c
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(1), m.c)
        self.assertIs(type(e.arg(0)), MonomialTermExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a * m.b
        e = m.c * e1
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.c)
        self.assertIs(type(e.arg(1)), MonomialTermExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a * m.b
        e2 = m.c * m.d
        e = e1 * e2
        self.assertIs(type(e), ProductExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), MonomialTermExpression)
        self.assertIs(type(e.arg(1)), ProductExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertIs(e.arg(1).arg(0), m.c)
        self.assertIs(e.arg(1).arg(1), m.d)
        self.assertEqual(e.size(), 7)

    def test_trivialProduct(self):
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(initialize=0)
        m.q = Param(initialize=1)
        e = m.a * 0
        self.assertExpressionsEqual(e, MonomialTermExpression((0, m.a)))
        e = 0 * m.a
        self.assertExpressionsEqual(e, MonomialTermExpression((0, m.a)))
        e = m.a * m.p
        self.assertExpressionsEqual(e, MonomialTermExpression((0, m.a)))
        e = m.p * m.a
        self.assertExpressionsEqual(e, MonomialTermExpression((0, m.a)))
        e = m.a * 1
        self.assertExpressionsEqual(e, m.a)
        e = 1 * m.a
        self.assertExpressionsEqual(e, m.a)
        e = m.a * m.q
        self.assertExpressionsEqual(e, m.a)
        e = m.q * m.a
        self.assertExpressionsEqual(e, m.a)
        e = NumericConstant(3) * NumericConstant(2)
        self.assertExpressionsEqual(e, 6)
        self.assertIs(type(e), int)
        self.assertEqual(e, 6)

    def test_simpleDivision(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a / m.b
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)

    def test_constDivision(self):
        m = AbstractModel()
        m.a = Var()
        e = m.a / 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertAlmostEqual(e.arg(0), 0.2)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)
        e = 5 / m.a
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(e.arg(1), m.a)
        self.assertEqual(e.size(), 3)

    def test_nestedDivision(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()
        e1 = 3 * m.b
        e = e1 / 5
        self.assertIs(type(e), MonomialTermExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 3.0 / 5)
        self.assertIs(e.arg(1), m.b)
        self.assertEqual(e.size(), 3)
        e1 = m.a / m.b
        e = e1 / 5
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(1), 5)
        self.assertIs(type(e.arg(0)), DivisionExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a / m.b
        e = 5 / e1
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 5)
        self.assertIs(type(e.arg(1)), DivisionExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a / m.b
        e = e1 / m.c
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(1), m.c)
        self.assertIs(type(e.arg(0)), DivisionExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a / m.b
        e = m.c / e1
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.c)
        self.assertIs(type(e.arg(1)), DivisionExpression)
        self.assertIs(e.arg(1).arg(0), m.a)
        self.assertIs(e.arg(1).arg(1), m.b)
        self.assertEqual(e.size(), 5)
        e1 = m.a / m.b
        e2 = m.c / m.d
        e = e1 / e2
        self.assertIs(type(e), DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), DivisionExpression)
        self.assertIs(type(e.arg(1)), DivisionExpression)
        self.assertIs(e.arg(0).arg(0), m.a)
        self.assertIs(e.arg(0).arg(1), m.b)
        self.assertIs(e.arg(1).arg(0), m.c)
        self.assertIs(e.arg(1).arg(1), m.d)
        self.assertEqual(e.size(), 7)

    def test_trivialDivision(self):
        m = AbstractModel()
        m.a = Var()
        m.p = Param()
        m.q = Param(initialize=2)
        m.r = Param(mutable=True)
        self.assertRaises(ZeroDivisionError, m.a.__div__, 0)
        e = 0 / m.a
        self.assertExpressionsEqual(e, DivisionExpression((0, m.a)))
        e = m.a / 1
        self.assertExpressionsEqual(e, m.a)
        e = 1 / m.a
        self.assertExpressionsEqual(e, DivisionExpression((1, m.a)))
        e = 1 / m.p
        self.assertIs(type(e), NPV_DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 1)
        self.assertIs(e.arg(1), m.p)
        e = 1 / m.q
        self.assertIs(type(e), NPV_DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 1)
        self.assertIs(e.arg(1), m.q)
        e = 1 / m.r
        self.assertIs(type(e), NPV_DivisionExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(e.arg(0), 1)
        self.assertIs(e.arg(1), m.r)
        e = NumericConstant(3) / NumericConstant(2)
        self.assertIs(type(e), float)
        self.assertEqual(e, 1.5)