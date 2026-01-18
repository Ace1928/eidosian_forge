import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
class TestTemplateSubstitution(unittest.TestCase):

    def setUp(self):
        self.m = m = ConcreteModel()
        m.TRAY = Set(initialize=range(5))
        m.TIME = Set(bounds=(0, 10), initialize=range(10))
        m.y = Var(initialize=1)
        m.x = Var(m.TIME, m.TRAY, initialize=lambda _m, i, j: i)
        m.dxdt = Var(m.TIME, m.TRAY, initialize=lambda _m, i, j: 2 * i)

    def test_simple_substitute_param(self):

        def diffeq(m, t, i):
            return m.dxdt[t, i] == t * m.x[t, i - 1] ** 2 + m.y ** 2 + m.x[t, i + 1] + m.x[t, i - 1]
        m = self.m
        t = IndexTemplate(m.TIME)
        e = diffeq(m, t, 2)
        self.assertTrue(isinstance(e, EXPR.RelationalExpression))
        _map = {}
        E = substitute_template_expression(e, substitute_getitem_with_param, _map)
        self.assertIsNot(e, E)
        self.assertEqual(len(_map), 3)
        idx1 = _GetItemIndexer(m.x[t, 1])
        self.assertEqual(idx1.nargs(), 2)
        self.assertIs(idx1.base, m.x)
        self.assertIs(idx1.arg(0), t)
        self.assertEqual(idx1.arg(1), 1)
        self.assertIn(idx1, _map)
        idx2 = _GetItemIndexer(m.dxdt[t, 2])
        self.assertEqual(idx2.nargs(), 2)
        self.assertIs(idx2.base, m.dxdt)
        self.assertIs(idx2.arg(0), t)
        self.assertEqual(idx2.arg(1), 2)
        self.assertIn(idx2, _map)
        idx3 = _GetItemIndexer(m.x[t, 3])
        self.assertEqual(idx3.nargs(), 2)
        self.assertIs(idx3.base, m.x)
        self.assertIs(idx3.arg(0), t)
        self.assertEqual(idx3.arg(1), 3)
        self.assertIn(idx3, _map)
        self.assertFalse(idx1 == idx2)
        self.assertFalse(idx1 == idx3)
        self.assertFalse(idx2 == idx3)
        idx4 = _GetItemIndexer(m.x[t, 2])
        self.assertNotIn(idx4, _map)
        t.set_value(5)
        self.assertEqual((e.arg(0)(), e.arg(1)()), (10, 136))
        self.assertEqual(str(E), "'dxdt[{TIME},2]'  ==  {TIME}*'x[{TIME},1]'**2 + y**2 + 'x[{TIME},3]' + 'x[{TIME},1]'")
        _map[idx1].set_value(value(m.x[value(t), 1]))
        _map[idx2].set_value(value(m.dxdt[value(t), 2]))
        _map[idx3].set_value(value(m.x[value(t), 3]))
        self.assertEqual((E.arg(0)(), E.arg(1)()), (10, 136))
        _map[idx1].set_value(12)
        _map[idx2].set_value(34)
        self.assertEqual((E.arg(0)(), E.arg(1)()), (34, 738))

    def test_simple_substitute_index(self):

        def diffeq(m, t, i):
            return m.dxdt[t, i] == t * m.x[t, i] ** 2 + m.y ** 2
        m = self.m
        t = IndexTemplate(m.TIME)
        e = diffeq(m, t, 2)
        t.set_value(5)
        self.assertTrue(isinstance(e, EXPR.RelationalExpression))
        self.assertEqual((e.arg(0)(), e.arg(1)()), (10, 126))
        E = substitute_template_expression(e, substitute_template_with_value)
        self.assertIsNot(e, E)
        self.assertEqual(str(E), 'dxdt[5,2]  ==  5.0*x[5,2]**2 + y**2')