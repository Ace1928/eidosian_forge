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
class TestNumericValue(unittest.TestCase):

    def test_asnum(self):
        try:
            as_numeric(None)
            self.fail('test_asnum - expected TypeError')
        except TypeError:
            pass

    def test_vals(self):
        a = NumericConstant(1.1)
        b = float(value(a))
        self.assertEqual(b, 1.1)
        b = int(value(a))
        self.assertEqual(b, 1)

    def test_ops(self):
        a = NumericConstant(1.1)
        b = NumericConstant(2.2)
        c = NumericConstant(-2.2)
        self.assertEqual(a() <= b(), True)
        self.assertEqual(a() >= b(), False)
        self.assertEqual(a() == b(), False)
        self.assertEqual(abs(a() + b() - 3.3) <= 1e-07, True)
        self.assertEqual(abs(b() - a() - 1.1) <= 1e-07, True)
        self.assertEqual(abs(b() * 3 - 6.6) <= 1e-07, True)
        self.assertEqual(abs(b() / 2 - 1.1) <= 1e-07, True)
        self.assertEqual(abs(abs(-b()) - 2.2) <= 1e-07, True)
        self.assertEqual(abs(c()), 2.2)
        self.assertEqual(str(c), '-2.2')

    def test_var(self):
        M = ConcreteModel()
        M.x = Var()
        e = M.x + 2
        self.assertRaises(ValueError, value, M.x)
        self.assertEqual(e(exception=False), None)