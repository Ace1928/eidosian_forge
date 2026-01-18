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
class TestLinearDecomp(unittest.TestCase):

    def setUp(self):
        pass

    def test_numeric(self):
        self.assertEqual(list(_decompose_linear_terms(2.0)), [(2.0, None)])

    def test_NPV(self):
        M = ConcreteModel()
        M.q = Param(initialize=2)
        self.assertEqual(list(_decompose_linear_terms(M.q)), [(M.q, None)])

    def test_var(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(list(_decompose_linear_terms(M.v)), [(1, M.v)])

    def test_simple(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(list(_decompose_linear_terms(2 * M.v)), [(2, M.v)])

    def test_sum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        self.assertEqual(list(_decompose_linear_terms(2 + M.v)), [(2, None), (1, M.v)])
        self.assertEqual(list(_decompose_linear_terms(M.q + M.v)), [(2, None), (1, M.v)])
        self.assertEqual(list(_decompose_linear_terms(M.v + M.q)), [(1, M.v), (2, None)])
        self.assertEqual(list(_decompose_linear_terms(M.w + M.v)), [(1, M.w), (1, M.v)])

    def test_prod(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        self.assertEqual(list(_decompose_linear_terms(2 * M.v)), [(2, M.v)])
        self.assertEqual(list(_decompose_linear_terms(M.q * M.v)), [(2, M.v)])
        self.assertEqual(list(_decompose_linear_terms(M.v * M.q)), [(2, M.v)])
        self.assertRaises(LinearDecompositionError, list, _decompose_linear_terms(M.w * M.v))

    def test_negation(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(list(_decompose_linear_terms(-M.v)), [(-1, M.v)])
        self.assertEqual(list(_decompose_linear_terms(-(2 + M.v))), [(-2, None), (-1, M.v)])

    def test_reciprocal(self):
        M = ConcreteModel()
        M.v = Var()
        M.q = Param(initialize=2)
        self.assertRaises(LinearDecompositionError, list, _decompose_linear_terms(1 / M.v))
        self.assertEqual(list(_decompose_linear_terms(1 / M.q)), [(0.5, None)])

    def test_multisum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        e = SumExpression([2])
        self.assertEqual(decompose_linear_term_wrapper(list(_decompose_linear_terms(e))), decompose_linear_term_wrapper([(2, None)]))
        e = SumExpression([2, M.v])
        self.assertEqual(decompose_linear_term_wrapper(list(_decompose_linear_terms(e))), decompose_linear_term_wrapper([(2, None), (1, M.v)]))
        e = SumExpression([2, M.q + M.v])
        self.assertEqual(decompose_linear_term_wrapper(list(_decompose_linear_terms(e))), decompose_linear_term_wrapper([(2, None), (2, None), (1, M.v)]))
        e = SumExpression([2, M.q + M.v, M.w])
        self.assertEqual(decompose_linear_term_wrapper(list(_decompose_linear_terms(e))), decompose_linear_term_wrapper([(2, None), (2, None), (1, M.v), (1, M.w)]))