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
class TestExpression_EvaluateNumericConstant(unittest.TestCase):

    def create(self, val, domain):
        return NumericConstant(val)

    def value_check(self, exp, val):
        """Check the value of the expression."""
        self.assertEqual(isinstance(exp, ExpressionBase), False)
        self.assertEqual(exp, val)

    def relation_check(self, exp, val):
        self.assertEqual(type(exp), bool)
        self.assertEqual(exp, val)

    def test_lt(self):
        a = self.create(1.3, Reals)
        b = self.create(2.0, Reals)
        self.relation_check(a < b, True)
        self.relation_check(a < a, False)
        self.relation_check(b < a, False)
        self.relation_check(a < 2.0, True)
        self.relation_check(a < 1.3, False)
        self.relation_check(b < 1.3, False)
        self.relation_check(1.3 < b, True)
        self.relation_check(1.3 < a, False)
        self.relation_check(2.0 < a, False)

    def test_gt(self):
        a = self.create(1.3, Reals)
        b = self.create(2.0, Reals)
        self.relation_check(a > b, False)
        self.relation_check(a > a, False)
        self.relation_check(b > a, True)
        self.relation_check(a > 2.0, False)
        self.relation_check(a > 1.3, False)
        self.relation_check(b > 1.3, True)
        self.relation_check(1.3 > b, False)
        self.relation_check(1.3 > a, False)
        self.relation_check(2.0 > a, True)

    def test_eq(self):
        a = self.create(1.3, Reals)
        b = self.create(2.0, Reals)
        self.relation_check(a == b, False)
        self.relation_check(a == a, True)
        self.relation_check(b == a, False)
        self.relation_check(a == 2.0, False)
        self.relation_check(a == 1.3, True)
        self.relation_check(b == 1.3, False)
        self.relation_check(1.3 == b, False)
        self.relation_check(1.3 == a, True)
        self.relation_check(2.0 == a, False)

    def test_arithmetic(self):
        a = self.create(-0.5, Reals)
        b = self.create(2.0, Reals)
        self.value_check(a - b, -2.5)
        self.value_check(a + b, 1.5)
        self.value_check(a * b, -1.0)
        self.value_check(b / a, -4.0)
        self.value_check(a ** b, 0.25)
        self.value_check(a - 2.0, -2.5)
        self.value_check(a + 2.0, 1.5)
        self.value_check(a * 2.0, -1.0)
        self.value_check(b / 0.5, 4.0)
        self.value_check(a ** 2.0, 0.25)
        self.value_check(0.5 - b, -1.5)
        self.value_check(0.5 + b, 2.5)
        self.value_check(0.5 * b, 1.0)
        self.value_check(2.0 / a, -4.0)
        self.value_check(0.5 ** b, 0.25)
        self.value_check(-a, 0.5)
        self.assertIs(+a, a)
        self.value_check(abs(-a), 0.5)