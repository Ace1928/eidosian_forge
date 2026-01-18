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
class TestPrettyPrinter_newStyle(unittest.TestCase):
    _save = None

    def setUp(self):
        TestPrettyPrinter_oldStyle._save = expr_common.TO_STRING_VERBOSE
        expr_common.TO_STRING_VERBOSE = False

    def tearDown(self):
        expr_common.TO_STRING_VERBOSE = TestPrettyPrinter_oldStyle._save

    def test_sum(self):
        model = ConcreteModel()
        model.a = Var()
        model.p = Param(mutable=True)
        expr = 5 + model.a + model.a
        self.assertIs(type(expr), LinearExpression)
        self.assertEqual('5 + a + a', str(expr))
        expr += 5
        self.assertIs(type(expr), LinearExpression)
        self.assertEqual('5 + a + a + 5', str(expr))
        expr = 2 + model.p
        self.assertEqual('2 + p', str(expr))
        expr = 2 - model.p
        self.assertEqual('2 - p', str(expr))

    def test_linearsum(self):
        model = ConcreteModel()
        A = range(5)
        model.a = Var(A)
        model.p = Param(A, initialize=2, mutable=True)
        expr = quicksum((i * model.a[i] for i in A)) + 3
        self.assertEqual('0*a[0] + a[1] + 2*a[2] + 3*a[3] + 4*a[4] + 3', str(expr))
        self.assertEqual('0*a[0] + a[1] + 2*a[2] + 3*a[3] + 4*a[4] + 3', expression_to_string(expr, compute_values=True))
        expr = quicksum(((i - 2) * model.a[i] for i in A)) + 3
        self.assertEqual('-2*a[0] - a[1] + 0*a[2] + a[3] + 2*a[4] + 3', str(expr))
        self.assertEqual('-2*a[0] - a[1] + 0*a[2] + a[3] + 2*a[4] + 3', expression_to_string(expr, compute_values=True))
        expr = quicksum((model.a[i] for i in A)) + 3
        self.assertEqual('a[0] + a[1] + a[2] + a[3] + a[4] + 3', str(expr))
        expr = quicksum((model.p[i] * model.a[i] for i in A))
        self.assertEqual('2*a[0] + 2*a[1] + 2*a[2] + 2*a[3] + 2*a[4]', expression_to_string(expr, compute_values=True))
        self.assertEqual('p[0]*a[0] + p[1]*a[1] + p[2]*a[2] + p[3]*a[3] + p[4]*a[4]', expression_to_string(expr, compute_values=False))
        self.assertEqual('p[0]*a[0] + p[1]*a[1] + p[2]*a[2] + p[3]*a[3] + p[4]*a[4]', str(expr))
        model.p[1].value = 0
        model.p[3].value = 3
        expr = quicksum((model.p[i] * model.a[i] if i != 3 else model.p[i] for i in A))
        self.assertEqual('2*a[0] + 0*a[1] + 2*a[2] + 3 + 2*a[4]', expression_to_string(expr, compute_values=True))
        expr = quicksum((model.p[i] * model.a[i] if i != 3 else -3 for i in A))
        self.assertEqual('p[0]*a[0] + p[1]*a[1] + p[2]*a[2] - 3 + p[4]*a[4]', expression_to_string(expr, compute_values=False))

    def test_negation(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        e = M.x * (1 + M.y)
        e = -e
        self.assertEqual('- x*(1 + y)', expression_to_string(e))
        M.x = -1
        M.x.fixed = True
        self.assertEqual('(1 + y)', expression_to_string(e, compute_values=True))

    def test_prod(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Var()
        expr = 5 * model.a * model.a
        self.assertEqual('5*a*a', str(expr))
        expr = 5 * model.a / model.a
        self.assertEqual('5*a/a', str(expr))
        expr = expr / model.a
        self.assertEqual('5*a/a/a', str(expr))
        expr = 5 * model.a / (model.a * model.a)
        self.assertEqual('5*a/(a*a)', str(expr))
        expr = 5 * model.a / model.a / 2
        self.assertEqual('5*a/a/2', str(expr))
        expr = model.a * model.b
        model.a = 1
        model.a.fixed = True
        self.assertEqual('b', expression_to_string(expr, compute_values=True))

    def test_inequality(self):
        model = ConcreteModel()
        model.a = Var()
        expr = 5 < model.a
        self.assertEqual('5  <  a', str(expr))
        expr = model.a >= 5
        self.assertEqual('5  <=  a', str(expr))
        expr = expr < 10
        self.assertEqual('5  <=  a  <  10', str(expr))
        expr = 5 <= model.a + 5
        self.assertEqual('5  <=  a + 5', str(expr))
        expr = expr < 10
        self.assertEqual('5  <=  a + 5  <  10', str(expr))

    def test_equality(self):
        model = ConcreteModel()
        model.a = Var()
        model.b = Param(initialize=5, mutable=True)
        expr = model.a == model.b
        self.assertEqual('a  ==  b', str(expr))
        expr = model.b == model.a
        self.assertEqual('b  ==  a', str(expr))
        expr = 5 == model.a
        self.assertEqual('a  ==  5', str(expr))
        expr = model.a == 10
        self.assertEqual('a  ==  10', str(expr))
        expr = 5 == model.a + 5
        self.assertEqual('a + 5  ==  5', str(expr))
        expr = model.a + 5 == 5
        self.assertEqual('a + 5  ==  5', str(expr))

    def test_linear(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.p = Param(initialize=2, mutable=True)
        expr = m.x - m.p * m.y
        self.assertEqual('x - p*y', str(expr))
        expr = m.x - m.p * m.y + 5
        self.assertIs(type(expr), LinearExpression)
        self.assertEqual('x - p*y + 5', str(expr))
        expr = m.x - m.p * m.y - 5
        self.assertIs(type(expr), LinearExpression)
        self.assertEqual('x - p*y - 5', str(expr))
        expr = m.x - m.p * m.y - 5 + m.p
        self.assertIs(type(expr), LinearExpression)
        self.assertEqual('x - p*y - 5 + p', str(expr))

    def test_expr_if(self):
        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        expr = Expr_if(IF_=m.a + m.b < 20, THEN_=m.a, ELSE_=m.b)
        self.assertEqual('Expr_if( ( a + b  <  20 ), then=( a ), else=( b ) )', str(expr))
        expr = Expr_if(IF=m.a + m.b < 20, THEN=1, ELSE=m.b)
        self.assertEqual('Expr_if( ( a + b  <  20 ), then=( 1 ), else=( b ) )', str(expr))
        with self.assertRaisesRegex(ValueError, 'Cannot specify both THEN_ and THEN'):
            Expr_if(IF_=m.a + m.b < 20, THEN_=1, ELSE_=m.b, THEN=2)
        with self.assertRaisesRegex(ValueError, 'Unrecognized arguments: _THEN_'):
            Expr_if(IF_=m.a + m.b < 20, _THEN_=1, ELSE_=m.b)

    def test_getitem(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 9)
        m.x = Var(m.I, initialize=lambda m, i: i + 1)
        m.P = Param(m.I, initialize=lambda m, i: 10 - i, mutable=True)
        t = IndexTemplate(m.I)
        e = m.x[t + m.P[t + 1]] + 3
        self.assertEqual('x[{I} + P[{I} + 1]] + 3', str(e))

    def test_associativity_rules(self):
        m = ConcreteModel()
        m.w = Var()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        self.assertEqual(str(m.z + m.x + m.y), 'z + x + y')
        self.assertEqual(str(m.z + m.x + m.y), 'z + x + y')
        self.assertEqual(str(m.w + m.z + (m.x + m.y)), 'w + z + x + y')
        self.assertEqual(str(m.z / m.x / (m.y / m.w)), 'z/x/(y/w)')
        self.assertEqual(str(m.z / m.x / m.y), 'z/x/y')
        self.assertEqual(str(m.z / m.x / m.y), 'z/x/y')
        self.assertEqual(str(m.z / (m.x / m.y)), 'z/(x/y)')
        self.assertEqual(str(m.z * m.x / m.y), 'z*x/y')
        self.assertEqual(str(m.z * m.x / m.y), 'z*x/y')
        self.assertEqual(str(m.z * (m.x / m.y)), 'z*(x/y)')
        self.assertEqual(str(m.z / m.x * m.y), 'z/x*y')
        self.assertEqual(str(m.z / m.x * m.y), 'z/x*y')
        self.assertEqual(str(m.z / (m.x * m.y)), 'z/(x*y)')
        self.assertEqual(str(m.x ** m.y ** m.z), 'x**(y**z)')
        self.assertEqual(str((m.x ** m.y) ** m.z), '(x**y)**z')
        self.assertEqual(str(m.x ** m.y ** m.z), 'x**(y**z)')

    def test_small_expression(self):
        model = AbstractModel()
        model.a = Var()
        model.b = Param(initialize=2, mutable=True)
        instance = model.create_instance()
        expr = instance.a + 1
        expr = expr - 1
        expr = expr * instance.a
        expr = expr / instance.a
        expr = expr ** instance.b
        expr = 1 - expr
        expr = 1 + expr
        expr = 2 * expr
        expr = 2 / expr
        expr = 2 ** expr
        expr = -expr
        expr = +expr
        expr = abs(expr)
        self.assertEqual('abs(- 2**(2/(2*(1 - ((a + 1 - 1)*a/a)**b + 1))))', str(expr))

    def test_large_expression(self):

        def c1_rule(model):
            return (1.0, model.b[1], None)

        def c2_rule(model):
            return (None, model.b[1], 0.0)

        def c3_rule(model):
            return (0.0, model.b[1], 1.0)

        def c4_rule(model):
            return (3.0, model.b[1])

        def c5_rule(model, i):
            return (model.b[i], 0.0)

        def c6a_rule(model):
            return 0.0 <= model.c

        def c7a_rule(model):
            return model.c <= 1.0

        def c7b_rule(model):
            return model.c >= 1.0

        def c8_rule(model):
            return model.c == 2.0

        def c9a_rule(model):
            return model.A + model.A <= model.c

        def c9b_rule(model):
            return model.A + model.A >= model.c

        def c10a_rule(model):
            return model.c <= model.B + model.B

        def c11_rule(model):
            return model.c == model.A + model.B

        def c15a_rule(model):
            return model.A <= model.A * model.d

        def c16a_rule(model):
            return model.A * model.d <= model.B

        def c12_rule(model):
            return model.c == model.d

        def c13a_rule(model):
            return model.c <= model.d

        def c14a_rule(model):
            return model.c >= model.d

        def cl_rule(model, i):
            if i > 10:
                return ConstraintList.End
            return i * model.c >= model.d

        def o2_rule(model, i):
            return model.b[i]
        model = AbstractModel()
        model.a = Set(initialize=[1, 2, 3])
        model.b = Var(model.a, initialize=1.1, within=PositiveReals)
        model.c = Var(initialize=2.1, within=PositiveReals)
        model.d = Var(initialize=3.1, within=PositiveReals)
        model.e = Var(initialize=4.1, within=PositiveReals)
        model.A = Param(default=-1, mutable=True)
        model.B = Param(default=-2, mutable=True)
        model.o2 = Objective(model.a, rule=o2_rule)
        model.o3 = Objective(model.a, model.a)
        model.c1 = Constraint(rule=c1_rule)
        model.c2 = Constraint(rule=c2_rule)
        model.c3 = Constraint(rule=c3_rule)
        model.c4 = Constraint(rule=c4_rule)
        model.c5 = Constraint(model.a, rule=c5_rule)
        model.c6a = Constraint(rule=c6a_rule)
        model.c7a = Constraint(rule=c7a_rule)
        model.c7b = Constraint(rule=c7b_rule)
        model.c8 = Constraint(rule=c8_rule)
        model.c9a = Constraint(rule=c9a_rule)
        model.c9b = Constraint(rule=c9b_rule)
        model.c10a = Constraint(rule=c10a_rule)
        model.c11 = Constraint(rule=c11_rule)
        model.c15a = Constraint(rule=c15a_rule)
        model.c16a = Constraint(rule=c16a_rule)
        model.c12 = Constraint(rule=c12_rule)
        model.c13a = Constraint(rule=c13a_rule)
        model.c14a = Constraint(rule=c14a_rule)
        model.cl = ConstraintList(rule=cl_rule)
        instance = model.create_instance()
        OUTPUT = open(join(currdir, 'varpprint.out'), 'w')
        instance.pprint(ostream=OUTPUT)
        OUTPUT.close()
        _out, _txt = (join(currdir, 'varpprint.out'), join(currdir, 'varpprint.txt'))
        self.assertTrue(cmp(_out, _txt), msg='Files %s and %s differ' % (_txt, _out))

    def test_labeler(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.z = Var()
        M.a = Var(range(3))
        M.p = Param(range(3), initialize=2)
        M.q = Param(range(3), initialize=3, mutable=True)
        e = M.x * M.y + sum_product(M.p, M.a) + quicksum((M.q[i] * M.a[i] for i in M.a)) / M.x
        self.assertEqual(str(e), 'x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (q[0]*a[0] + q[1]*a[1] + q[2]*a[2])/x')
        self.assertEqual(e.to_string(), 'x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (q[0]*a[0] + q[1]*a[1] + q[2]*a[2])/x')
        self.assertEqual(e.to_string(compute_values=True), 'x*y + (2*a[0] + 2*a[1] + 2*a[2]) + (3*a[0] + 3*a[1] + 3*a[2])/x')
        labeler = NumericLabeler('x')
        self.assertEqual(expression_to_string(e, labeler=labeler), 'x1*x2 + (2*x3 + 2*x4 + 2*x5) + (x6*x3 + x7*x4 + x8*x5)/x1')
        from pyomo.core.expr.symbol_map import SymbolMap
        labeler = NumericLabeler('x')
        smap = SymbolMap(labeler)
        self.assertEqual(expression_to_string(e, smap=smap), 'x1*x2 + (2*x3 + 2*x4 + 2*x5) + (x6*x3 + x7*x4 + x8*x5)/x1')
        self.assertEqual(expression_to_string(e, smap=smap, compute_values=True), 'x1*x2 + (2*x3 + 2*x4 + 2*x5) + (3*x3 + 3*x4 + 3*x5)/x1')

    def test_balanced_parens(self):
        self.assertTrue(_balanced_parens('(1+5)+((x - 1)*(5+x))'))
        self.assertFalse(_balanced_parens('1+5)+((x - 1)*(5+x)'))
        self.assertFalse(_balanced_parens('(((1+5)+((x - 1)*(5+x))'))
        self.assertFalse(_balanced_parens('1+5)+((x - 1)*(5+x))'))
        self.assertFalse(_balanced_parens('(1+5)+((x - 1)*(5+x)'))
        self.assertFalse(_balanced_parens('(1+5)+((x - 1))*(5+x))'))