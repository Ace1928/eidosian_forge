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
class Test_decompose_linear_terms(unittest.TestCase):

    def test_numeric(self):
        self.assertEqual(decompose_term(2.0), (True, [(2.0, None)]))

    def test_NPV(self):
        M = ConcreteModel()
        M.q = Param(initialize=2)
        self.assertEqual(decompose_term(M.q), (True, [(M.q, None)]))

    def test_var(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(decompose_term(M.v), (True, [(1, M.v)]))

    def test_simple(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(decompose_term(2 * M.v), (True, [(2, M.v)]))

    def test_sum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        self.assertEqual(decompose_term(2 + M.v), (True, [(2, None), (1, M.v)]))
        self.assertEqual(decompose_term(M.q + M.v), (True, [(2, None), (1, M.v)]))
        self.assertEqual(decompose_term(M.v + M.q), (True, [(1, M.v), (2, None)]))
        self.assertEqual(decompose_term(M.v + M.w), (True, [(1, M.v), (1, M.w)]))

    def test_prod(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        self.assertEqual(decompose_term(2 * M.v), (True, [(2, M.v)]))
        self.assertEqual(decompose_term(M.q * M.v), (True, [(2, M.v)]))
        self.assertEqual(decompose_term(M.v * M.q), (True, [(2, M.v)]))
        self.assertEqual(decompose_term(M.w * M.v), (False, None))

    def test_negation(self):
        M = ConcreteModel()
        M.v = Var()
        self.assertEqual(decompose_term(-M.v), (True, [(-1, M.v)]))
        self.assertEqual(decompose_term(-(2 + M.v)), (True, [(-2, None), (-1, M.v)]))

    def test_reciprocal(self):
        M = ConcreteModel()
        M.v = Var()
        M.q = Param(initialize=2)
        M.p = Param(initialize=2, mutable=True)
        self.assertEqual(decompose_term(1 / M.v), (False, None))
        self.assertEqual(decompose_term(1 / M.q), (True, [(0.5, None)]))
        e = 1 / M.p
        self.assertEqual(decompose_term(e), (True, [(e, None)]))

    def test_multisum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=3)
        e = SumExpression([2])
        self.assertEqual(decompose_term_wrapper(decompose_term(e)), decompose_term_wrapper((True, [(2, None)])))
        e = SumExpression([2, M.v])
        self.assertEqual(decompose_term_wrapper(decompose_term(e)), decompose_term_wrapper((True, [(2, None), (1, M.v)])))
        e = SumExpression([2, M.q + M.v])
        self.assertEqual(decompose_term_wrapper(decompose_term(e)), decompose_term_wrapper((True, [(2, None), (3, None), (1, M.v)])))
        e = SumExpression([2, M.q + M.v, M.w])
        self.assertEqual(decompose_term_wrapper(decompose_term(e)), decompose_term_wrapper((True, [(2, None), (3, None), (1, M.v), (1, M.w)])))

    def test_linear(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        with linear_expression() as e:
            e += 2
            self.assertEqual(decompose_term(e), (True, [(2, None)]))
            e += M.v
            self.assertEqual(decompose_term(-e), (True, [(-2, None), (-1, M.v)]))