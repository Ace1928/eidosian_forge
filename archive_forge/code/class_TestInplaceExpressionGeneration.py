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
class TestInplaceExpressionGeneration(unittest.TestCase):

    def setUp(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        self.m = m

    def tearDown(self):
        self.m = None

    def test_iadd(self):
        m = self.m
        x = 0
        x += m.a
        self.assertIs(type(x), type(m.a))
        x += m.a
        self.assertIs(type(x), LinearExpression)
        self.assertEqual(x.nargs(), 2)
        x += m.b
        self.assertIs(type(x), LinearExpression)
        self.assertEqual(x.nargs(), 3)

    def test_isub(self):
        m = self.m
        x = m.a
        x -= 0
        self.assertIs(type(x), type(m.a))
        x = 0
        x -= m.a
        self.assertIs(type(x), MonomialTermExpression)
        self.assertEqual(x.nargs(), 2)
        x -= m.a
        self.assertIs(type(x), LinearExpression)
        self.assertEqual(x.nargs(), 2)
        x -= m.a
        self.assertIs(type(x), LinearExpression)
        self.assertEqual(x.nargs(), 3)
        x -= m.b
        self.assertIs(type(x), LinearExpression)
        self.assertEqual(x.nargs(), 4)

    def test_imul(self):
        m = self.m
        x = 1
        x *= m.a
        self.assertIs(type(x), type(m.a))
        x *= m.a
        self.assertIs(type(x), ProductExpression)
        self.assertEqual(x.nargs(), 2)
        x *= m.a
        self.assertIs(type(x), ProductExpression)
        self.assertEqual(x.nargs(), 2)

    def test_idiv(self):
        m = self.m
        x = 1
        x /= m.a
        self.assertIs(type(x), DivisionExpression)
        self.assertEqual(x.arg(0), 1)
        self.assertIs(x.arg(1), m.a)
        x /= m.a
        self.assertIs(type(x), DivisionExpression)
        self.assertIs(type(x.arg(0)), DivisionExpression)
        self.assertIs(x.arg(0).arg(1), m.a)
        self.assertIs(x.arg(1), m.a)

    def test_ipow(self):
        m = self.m
        x = 1
        x **= m.a
        self.assertIs(type(x), PowExpression)
        self.assertEqual(x.nargs(), 2)
        self.assertEqual(value(x.arg(0)), 1)
        self.assertIs(x.arg(1), m.a)
        x **= m.b
        self.assertIs(type(x), PowExpression)
        self.assertEqual(x.nargs(), 2)
        self.assertIs(type(x.arg(0)), PowExpression)
        self.assertIs(x.arg(1), m.b)
        self.assertEqual(x.nargs(), 2)
        self.assertEqual(value(x.arg(0).arg(0)), 1)
        self.assertIs(x.arg(0).arg(1), m.a)
        x = 1 ** m.a
        y = x
        x **= m.b
        self.assertIs(type(y), PowExpression)
        self.assertEqual(y.nargs(), 2)
        self.assertEqual(value(y.arg(0)), 1)
        self.assertIs(y.arg(1), m.a)
        self.assertIs(type(x), PowExpression)
        self.assertEqual(x.nargs(), 2)
        self.assertIs(type(x.arg(0)), PowExpression)
        self.assertIs(x.arg(1), m.b)
        self.assertEqual(x.nargs(), 2)
        self.assertEqual(value(x.arg(0).arg(0)), 1)
        self.assertIs(x.arg(0).arg(1), m.a)