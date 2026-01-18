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
class EntangledExpressionErrors(unittest.TestCase):

    def test_sumexpr_add_entangled(self):
        x = Var()
        e = x * 2 + 1
        e + 1

    def test_entangled_test1(self):
        self.m = ConcreteModel()
        self.m.a = Var()
        self.m.b = Var()
        self.m.c = Var()
        self.m.d = Var()
        e1 = self.m.a + self.m.b
        e2 = self.m.c + e1
        e3 = self.m.d + e1
        self.assertEqual(e1.nargs(), 2)
        self.assertEqual(e2.nargs(), 3)
        self.assertEqual(e3.nargs(), 3)
        self.assertNotEqual(id(e2.arg(2)), id(e3.arg(2)))