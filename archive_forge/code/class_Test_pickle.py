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
class Test_pickle(unittest.TestCase):

    def test_simple(self):
        M = ConcreteModel()
        M.v = Var()
        e = 2 * M.v
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_sum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        e = M.v + M.q
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def Xtest_Sum(self):
        M = ConcreteModel()
        A = range(5)
        M.v = Var(A)
        e = quicksum((M.v[i] for i in M.v))
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_prod(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=2)
        e = M.v * M.q
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_negation(self):
        M = ConcreteModel()
        M.v = Var()
        e = -(2 + M.v)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_reciprocal(self):
        M = ConcreteModel()
        M.v = Var()
        M.q = Param(initialize=2)
        M.p = Param(initialize=2, mutable=True)
        e = 1 / M.p
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_multisum(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        M.q = Param(initialize=3)
        e = SumExpression([2, M.q + M.v, M.w])
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_linear(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        e = LinearExpression()
        e += 2
        e += M.v
        e = -e
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_linear_context(self):
        M = ConcreteModel()
        M.v = Var()
        M.w = Var()
        with linear_expression() as e:
            e += 2
            e += M.v
        e = -e
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_ExprIf(self):
        M = ConcreteModel()
        M.v = Var()
        e = Expr_if(M.v, 1, 0)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)

    def test_getitem(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 9)
        m.x = Var(m.I, initialize=x_)
        m.P = Param(m.I, initialize=P_, mutable=True)
        t = IndexTemplate(m.I)
        e = m.x[t + m.P[t + 1]] + 3
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)
        self.assertEqual('x[{I} + P[{I} + 1]] + 3', str(e))

    def test_abs(self):
        M = ConcreteModel()
        M.v = Var()
        e = abs(M.v)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)
        self.assertEqual(str(e), str(e_))

    def test_sin(self):
        M = ConcreteModel()
        M.v = Var()
        e = sin(M.v)
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)
        self.assertEqual(str(e), str(e_))

    def test_external_fcn(self):
        model = ConcreteModel()
        model.a = Var()
        model.x = ExternalFunction(library='foo.so', function='bar')
        e = model.x(model.a, 1, 'foo', [])
        s = pickle.dumps(e)
        e_ = pickle.loads(s)
        self.assertIsNot(e, e_)
        self.assertExpressionsStructurallyEqual(e, e_)