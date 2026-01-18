import pickle
import os
import io
import sys
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import value
from pyomo.core.expr.relational_expr import (
class TestGenerate_RangedExpression(unittest.TestCase):

    def setUp(self):
        m = AbstractModel()
        m.I = Set()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.x = Var(m.I)
        self.m = m

    def tearDown(self):
        self.m = None

    def test_compoundInequality(self):
        m = self.m
        e = inequality(m.a, m.b, m.c, strict=True)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), m.c)
        self.assertEqual(e._strict[0], True)
        self.assertEqual(e._strict[1], True)
        e = inequality(m.a, m.b, m.c)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(0), m.a)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(2), m.c)
        self.assertEqual(e._strict[0], False)
        self.assertEqual(e._strict[1], False)
        e = inequality(upper=m.c, body=m.b, lower=m.a, strict=True)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(2), m.c)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(0), m.a)
        self.assertEqual(e._strict[0], True)
        self.assertEqual(e._strict[1], True)
        e = inequality(upper=m.c, body=m.b, lower=m.a)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(2), m.c)
        self.assertIs(e.arg(1), m.b)
        self.assertIs(e.arg(0), m.a)
        self.assertEqual(e._strict[0], False)
        self.assertEqual(e._strict[1], False)
        e = inequality(0, m.a, 0)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(2), 0)
        self.assertIs(e.arg(1), m.a)
        self.assertIs(e.arg(0), 0)
        self.assertEqual(e._strict[0], False)
        self.assertEqual(e._strict[1], False)
        e = inequality(0, m.a, 0, True)
        self.assertIs(type(e), RangedExpression)
        self.assertEqual(e.nargs(), 3)
        self.assertIs(e.arg(2), 0)
        self.assertIs(e.arg(1), m.a)
        self.assertIs(e.arg(0), 0)
        self.assertEqual(e._strict[0], True)
        self.assertEqual(e._strict[1], True)

    def test_val1(self):
        m = ConcreteModel()
        m.v = Var(initialize=2)
        e = inequality(0, m.v, 2)
        self.assertEqual(value(e), True)
        e = inequality(0, m.v, 1)
        self.assertEqual(value(e), False)
        e = inequality(0, m.v, 2, strict=True)
        self.assertEqual(value(e), False)

    def test_val2(self):
        m = ConcreteModel()
        m.v = Var(initialize=2)
        e = 1 < m.v
        e = e <= 2
        self.assertEqual(value(e), True)
        e = 1 <= m.v
        e = e < 2
        self.assertEqual(value(e), False)