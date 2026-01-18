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
class TestLinearExpression(unittest.TestCase):

    def test_init(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        e = LinearExpression(constant=5, linear_vars=[m.x, m.y], linear_coefs=[2, 3])
        self.assertEqual(e.constant, 5)
        self.assertEqual(e.linear_vars, [m.x, m.y])
        self.assertEqual(e.linear_coefs, [2, 3])
        f = LinearExpression([5, 2 * m.x, 3 * m.y])
        self.assertEqual(e.constant, 5)
        self.assertEqual(e.linear_vars, [m.x, m.y])
        self.assertEqual(e.linear_coefs, [2, 3])
        self.assertExpressionsEqual(e, f)
        args = [10, MonomialTermExpression((4, m.y)), MonomialTermExpression((5, m.x))]
        with LoggingIntercept() as OUT:
            e = LinearExpression(args)
        self.assertEqual(OUT.getvalue(), '')
        self.assertIs(e._args_, args)
        self.assertEqual(e.constant, 10)
        self.assertEqual(e.linear_vars, [m.y, m.x])
        self.assertEqual(e.linear_coefs, [4, 5])

    def test_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        e = LinearExpression()
        self.assertEqual(e.to_string(), '0')
        e = LinearExpression(constant=0, linear_coefs=[-1, 1, -2, 2], linear_vars=[m.x, m.y, m.x, m.y])
        self.assertEqual(e.to_string(), '- x + y - 2*x + 2*y')
        e = LinearExpression(constant=10, linear_coefs=[-1, 1, -2, 2], linear_vars=[m.x, m.y, m.x, m.y])
        self.assertEqual(e.to_string(), '10 - x + y - 2*x + 2*y')

    def test_sum_other(self):
        m = ConcreteModel()
        m.v = Var(range(5))
        m.p = Param(mutable=True, initialize=4)
        for arg in (2, m.p):
            with linear_expression() as e:
                e += arg
                self.assertIs(e.__class__, _MutableNPVSumExpression)
                e -= arg
                self.assertIs(e.__class__, _MutableNPVSumExpression)
        for arg in (m.v[0], m.p * m.v[0]):
            with linear_expression() as e:
                e += arg
                self.assertIs(e.__class__, _MutableLinearExpression)
                e -= arg
                self.assertIs(e.__class__, _MutableLinearExpression)
        arg = 1 + m.v[0]
        with linear_expression() as e:
            e += arg
            self.assertIs(e.__class__, _MutableLinearExpression)
            e -= arg
            self.assertIs(e.__class__, _MutableSumExpression)
        for arg in (m.p * (1 + m.v[0]), m.v[0] * m.v[1]):
            with linear_expression() as e:
                e += arg
                self.assertIs(e.__class__, _MutableSumExpression)
                self.assertIs(e.args[-1], arg)
            with linear_expression() as e:
                e -= arg
                self.assertIs(e.__class__, _MutableSumExpression)
                self.assertIs(e.args[-1].__class__, NegationExpression)
                self.assertIs(e.args[-1].arg(0), arg)
        for arg in (2, m.p, m.v[0], m.p * m.v[0], 1 + m.v[0], m.p * (1 + m.v[0]), m.v[0] * m.v[1]):
            with linear_expression() as e:
                e = e + arg
                self.assertIs(e, arg)
            with linear_expression() as e:
                e = arg + e
                self.assertIs(e, arg)
            with linear_expression() as e:
                e = arg - e
                self.assertIs(e, arg)
            with linear_expression() as e:
                e = e - arg
                self.assertExpressionsEqual(e, -arg)

    def test_mul_other(self):
        m = ConcreteModel()
        m.v = Var(range(5), initialize=1)
        m.p = Param(initialize=2, mutable=True)
        with linear_expression() as e:
            e += 1
            self.assertIs(e.__class__, _MutableNPVSumExpression)
            self.assertEqual('1', str(e))
            f = 2 * e
            self.assertEqual(f, 2)
            self.assertIs(e.__class__, NPV_SumExpression)
            self.assertIs(f.__class__, int)
        with linear_expression() as e:
            e += 1 + m.v[0]
            self.assertIs(e.__class__, _MutableLinearExpression)
            f = e * 2
            self.assertIs(e.__class__, LinearExpression)
            self.assertIs(f.__class__, ProductExpression)
        with linear_expression() as e:
            e += 1
            self.assertIs(e.__class__, _MutableNPVSumExpression)
            f = e * m.p
            self.assertEqual('p', str(f))
            self.assertIs(e.__class__, NPV_SumExpression)
            self.assertIs(f, m.p)
        with linear_expression() as e:
            e += 1
            self.assertIs(e.__class__, _MutableNPVSumExpression)
            f = e * 0
            self.assertIs(e.__class__, NPV_SumExpression)
            self.assertEqual(f, 0)
        with linear_expression() as e:
            e += m.v[0]
            self.assertIs(e.__class__, _MutableLinearExpression)
            f = e * 2
            self.assertEqual('v[0]', str(e))
            self.assertEqual('2*v[0]', str(f))
            self.assertIs(e.__class__, LinearExpression)
            self.assertIs(f.__class__, MonomialTermExpression)
        with linear_expression() as e:
            e += 1
            self.assertIs(e.__class__, _MutableNPVSumExpression)
            e *= m.v[0] * m.v[1]
            self.assertIs(e.__class__, ProductExpression)
        with linear_expression() as e:
            e += 1
            self.assertIs(e.__class__, _MutableNPVSumExpression)
            f = e * (m.v[0] * m.v[1])
            self.assertIs(e.__class__, NPV_SumExpression)
            self.assertIs(f.__class__, ProductExpression)
        with linear_expression() as e:
            e += 1
            self.assertIs(e.__class__, _MutableNPVSumExpression)
            f = m.v[0] * m.v[1] * e
            self.assertIs(e.__class__, NPV_SumExpression)
            self.assertIs(f.__class__, ProductExpression)

    def test_div(self):
        m = ConcreteModel()
        m.v = Var(range(5), initialize=1)
        m.p = Param(initialize=2, mutable=True)
        with linear_expression() as e:
            e += m.v[0]
            self.assertIs(e.__class__, _MutableLinearExpression)
            e /= 2
            self.assertEqual('0.5*v[0]', str(e))
            self.assertIs(e.__class__, MonomialTermExpression)
        with linear_expression() as e:
            e += m.v[0]
            self.assertIs(e.__class__, _MutableLinearExpression)
            e /= m.p
            self.assertEqual('1/p*v[0]', str(e))
            self.assertIs(e.__class__, MonomialTermExpression)
        with linear_expression() as e:
            e += 1
            self.assertIs(e.__class__, _MutableNPVSumExpression)
            e /= m.v[0]
            self.assertIs(e.__class__, DivisionExpression)

    def test_div_other(self):
        m = ConcreteModel()
        m.v = Var(range(5), initialize=1)
        m.p = Param(initialize=2, mutable=True)
        with linear_expression() as e:
            e += m.v[0]
            try:
                e = 1 / e
                self.fail('Expected ValueError')
            except:
                pass
        with linear_expression() as e:
            e += 1
            e = 1 / e
            self.assertEqual('1.0', str(e))

    def test_negation_other(self):
        m = ConcreteModel()
        m.v = Var(range(5))
        with linear_expression() as e:
            e += 2
            e += m.v[1]
            self.assertIs(e.__class__, _MutableLinearExpression)
            e = -e
            self.assertIs(e.__class__, NegationExpression)
            self.assertIs(e.arg(0).__class__, LinearExpression)

    def test_pow_other(self):
        m = ConcreteModel()
        m.v = Var(range(5))
        m.p = Param(initialize=5, mutable=True)
        with linear_expression() as e:
            e = 2 ** e
            self.assertIs(e, 1)
        with linear_expression() as e:
            e += 2
            e = 2 ** e
            self.assertIs(e, 4)
        with linear_expression() as e:
            e += m.p
            e = 2 ** e
            self.assertExpressionsEqual(e, NPV_PowExpression((2, m.p)))
        with linear_expression() as e:
            e += m.v[0] + m.v[1]
            e = m.v[0] ** e
            self.assertExpressionsEqual(e, PowExpression((m.v[0], LinearExpression([MonomialTermExpression((1, m.v[0])), MonomialTermExpression((1, m.v[1]))]))))