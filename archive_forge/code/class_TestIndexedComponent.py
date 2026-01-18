import os
from os.path import abspath, dirname
from pyomo.common import DeveloperError
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Var, Param, Set, value, Integers
from pyomo.core.base.set import FiniteSetOf, OrderedSetOf
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.expr import GetItemExpression
from pyomo.core import SortComponents
class TestIndexedComponent(unittest.TestCase):

    def test_normalize_index(self):
        self.assertEqual('abc', normalize_index('abc'))
        self.assertEqual(1, normalize_index(1))
        self.assertEqual(1, normalize_index([1]))
        self.assertEqual((1, 2, 3), normalize_index((1, 2, 3)))
        self.assertEqual((1, 2, 3), normalize_index([1, 2, 3]))
        self.assertEqual((1, 2, 3, 4), normalize_index((1, 2, [3, 4])))
        self.assertEqual((1, 2, 'abc'), normalize_index((1, 2, 'abc')))
        self.assertEqual((1, 2, 'abc'), normalize_index((1, 2, ('abc',))))
        a = [0, 9, 8]
        self.assertEqual((1, 2, 0, 9, 8), normalize_index((1, 2, a)))
        self.assertEqual((1, 2, 3, 4, 5), normalize_index([[], 1, [], 2, [[], 3, [[], 4, []], []], 5, []]))
        self.assertEqual((), normalize_index([[[[], []], []], []]))
        self.assertEqual((), normalize_index([[], [[], [[]]]]))
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1])
        m.i = Set(initialize=[1])
        m.j = Set([1], initialize=[1])
        self.assertIs(m, normalize_index(m))
        self.assertIs(m.x, normalize_index(m.x))
        self.assertIs(m.y, normalize_index(m.y))
        self.assertIs(m.y[1], normalize_index(m.y[1]))
        self.assertIs(m.i, normalize_index(m.i))
        self.assertIs(m.j, normalize_index(m.j))
        self.assertIs(m.j[1], normalize_index(m.j[1]))

    def test_index_by_constant_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2)
        m.x = Var([1, 2, 3], initialize=lambda m, x: 2 * x)
        self.assertEqual(value(m.x[2]), 4)
        self.assertEqual(value(m.x[m.i]), 4)
        self.assertIs(m.x[2], m.x[m.i])

    def test_index_by_multiple_constant_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2)
        m.j = Param(initialize=3)
        m.x = Var([1, 2, 3], [1, 2, 3], initialize=lambda m, x, y: 2 * x * y)
        self.assertEqual(value(m.x[2, 3]), 12)
        self.assertEqual(value(m.x[m.i, 3]), 12)
        self.assertEqual(value(m.x[m.i, m.j]), 12)
        self.assertEqual(value(m.x[2, m.j]), 12)
        self.assertIs(m.x[2, 3], m.x[m.i, 3])
        self.assertIs(m.x[2, 3], m.x[m.i, m.j])
        self.assertIs(m.x[2, 3], m.x[2, m.j])

    def test_index_by_fixed_simpleComponent(self):
        m = ConcreteModel()
        m.i = Param(initialize=2, mutable=True)
        m.x = Var([1, 2, 3], initialize=lambda m, x: 2 * x)
        self.assertEqual(value(m.x[2]), 4)
        self.assertRaisesRegex(RuntimeError, 'is a fixed but not constant value', m.x.__getitem__, m.i)

    def test_index_by_variable_simpleComponent(self):
        m = ConcreteModel()
        m.i = Var(initialize=2, domain=Integers)
        m.x = Var([1, 2, 3], initialize=lambda m, x: 2 * x)
        self.assertEqual(value(m.x[2]), 4)
        thing = m.x[m.i]
        self.assertIsInstance(thing, GetItemExpression)
        self.assertEqual(len(thing.args), 2)
        self.assertIs(thing.args[0], m.x)
        self.assertIs(thing.args[1], m.i)
        idx_expr = 2 * m.i + 1
        thing = m.x[idx_expr]
        self.assertIsInstance(thing, GetItemExpression)
        self.assertEqual(len(thing.args), 2)
        self.assertIs(thing.args[0], m.x)
        self.assertIs(thing.args[1], idx_expr)

    def test_index_param_by_variable(self):
        m = ConcreteModel()
        m.i = Var(initialize=2, domain=Integers)
        m.p = Param([1, 2, 3], initialize=lambda m, x: 2 * x)
        thing = m.p[m.i]
        self.assertIsInstance(thing, GetItemExpression)
        self.assertEqual(len(thing.args), 2)
        self.assertIs(thing.args[0], m.p)
        self.assertIs(thing.args[1], m.i)
        idx_expr = 2 ** m.i + 1
        thing = m.p[idx_expr]
        self.assertIsInstance(thing, GetItemExpression)
        self.assertEqual(len(thing.args), 2)
        self.assertIs(thing.args[0], m.p)
        self.assertIs(thing.args[1], idx_expr)

    def test_index_var_by_tuple_with_variables(self):
        m = ConcreteModel()
        m.x = Var([(1, 1), (2, 1), (1, 2), (2, 2)])
        m.i = Var([1, 2, 3], domain=Integers)
        thing = m.x[1, m.i[1]]
        self.assertIsInstance(thing, GetItemExpression)
        self.assertEqual(len(thing.args), 3)
        self.assertIs(thing.args[0], m.x)
        self.assertEqual(thing.args[1], 1)
        self.assertIs(thing.args[2], m.i[1])
        idx_expr = m.i[1] + m.i[2] * m.i[3]
        thing = m.x[1, idx_expr]
        self.assertIsInstance(thing, GetItemExpression)
        self.assertEqual(len(thing.args), 3)
        self.assertIs(thing.args[0], m.x)
        self.assertEqual(thing.args[1], 1)
        self.assertIs(thing.args[2], idx_expr)

    def test_index_by_unhashable_type(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3], initialize=lambda m, x: 2 * x)
        self.assertRaisesRegex(TypeError, '.*', m.x.__getitem__, {})
        self.assertIs(m.x[[1]], m.x[1])
        m.y = Var([(1, 1), (1, 2)])
        self.assertIs(m.y[[1, 1]], m.y[1, 1])
        m.y[[1, 2]] = 5
        y12 = m.y[[1, 2]]
        self.assertEqual(y12.value, 5)
        m.y[[1, 2]] = 15
        self.assertIs(y12, m.y[[1, 2]])
        self.assertEqual(y12.value, 15)
        with self.assertRaisesRegex(KeyError, "Index '\\(2, 2\\)' is not valid for indexed component 'y'"):
            m.y[[2, 2]] = 5

    def test_ordered_keys(self):
        m = ConcreteModel()
        init_keys = [2, 1, (1, 2), (1, 'a'), (1, 1)]
        m.I = Set(ordered=False, dimen=None, initialize=init_keys)
        ordered_keys = [1, 2, (1, 1), (1, 2), (1, 'a')]
        m.x = Var(m.I)
        self.assertNotEqual(list(m.x.keys()), list(m.x.keys(True)))
        self.assertEqual(set(m.x.keys()), set(m.x.keys(True)))
        self.assertEqual(ordered_keys, list(m.x.keys(True)))
        m.P = Param(m.I, initialize={k: v for v, k in enumerate(init_keys)})
        self.assertNotEqual(list(m.P.keys()), list(m.P.keys(True)))
        self.assertEqual(set(m.P.keys()), set(m.P.keys(True)))
        self.assertEqual(ordered_keys, list(m.P.keys(True)))
        self.assertEqual([1, 0, 4, 2, 3], list(m.P.values(True)))
        self.assertEqual(list(zip(ordered_keys, [1, 0, 4, 2, 3])), list(m.P.items(True)))
        m.P = Param(m.I, initialize={(1, 2): 30, 1: 10, 2: 20}, default=1)
        self.assertNotEqual(list(m.P.keys()), list(m.P.keys(True)))
        self.assertEqual(set(m.P.keys()), set(m.P.keys(True)))
        self.assertEqual(ordered_keys, list(m.P.keys(True)))
        self.assertEqual([10, 20, 1, 30, 1], list(m.P.values(True)))
        self.assertEqual(list(zip(ordered_keys, [10, 20, 1, 30, 1])), list(m.P.items(True)))

    def test_ordered_keys_deprecation(self):
        m = ConcreteModel()
        unordered = [1, 3, 2]
        ordered = [1, 2, 3]
        m.I = FiniteSetOf(unordered)
        m.x = Var(m.I)
        self.assertEqual(list(m.x.keys()), unordered)
        self.assertEqual(list(m.x.keys(SortComponents.ORDERED_INDICES)), ordered)
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.keys(True)), ordered)
        self.assertEqual(LOG.getvalue(), '')
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.keys(ordered=True)), ordered)
        self.assertIn('keys(ordered=True) is deprecated', LOG.getvalue())
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.keys(ordered=False)), unordered)
        self.assertIn('keys(ordered=False) is deprecated', LOG.getvalue())
        m = ConcreteModel()
        unordered = [1, 3, 2]
        ordered = [1, 2, 3]
        m.I = OrderedSetOf(unordered)
        m.x = Var(m.I)
        self.assertEqual(list(m.x.keys()), unordered)
        self.assertEqual(list(m.x.keys(SortComponents.ORDERED_INDICES)), unordered)
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.keys(True)), ordered)
        self.assertEqual(LOG.getvalue(), '')
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.keys(ordered=True)), unordered)
        self.assertIn('keys(ordered=True) is deprecated', LOG.getvalue())
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.keys(ordered=False)), unordered)
        self.assertIn('keys(ordered=False) is deprecated', LOG.getvalue())

    def test_ordered_values_deprecation(self):
        m = ConcreteModel()
        unordered = [1, 3, 2]
        ordered = [1, 2, 3]
        m.I = FiniteSetOf(unordered)
        m.x = Var(m.I)
        unordered = [m.x[i] for i in unordered]
        ordered = [m.x[i] for i in ordered]
        self.assertEqual(list(m.x.values()), unordered)
        self.assertEqual(list(m.x.values(SortComponents.ORDERED_INDICES)), ordered)
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.values(True)), ordered)
        self.assertEqual(LOG.getvalue(), '')
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.values(ordered=True)), ordered)
        self.assertIn('values(ordered=True) is deprecated', LOG.getvalue())
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.values(ordered=False)), unordered)
        self.assertIn('values(ordered=False) is deprecated', LOG.getvalue())
        m = ConcreteModel()
        unordered = [1, 3, 2]
        ordered = [1, 2, 3]
        m.I = OrderedSetOf(unordered)
        m.x = Var(m.I)
        unordered = [m.x[i] for i in unordered]
        ordered = [m.x[i] for i in ordered]
        self.assertEqual(list(m.x.values()), unordered)
        self.assertEqual(list(m.x.values(SortComponents.ORDERED_INDICES)), unordered)
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.values(True)), ordered)
        self.assertEqual(LOG.getvalue(), '')
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.values(ordered=True)), unordered)
        self.assertIn('values(ordered=True) is deprecated', LOG.getvalue())
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.values(ordered=False)), unordered)
        self.assertIn('values(ordered=False) is deprecated', LOG.getvalue())

    def test_ordered_items_deprecation(self):
        m = ConcreteModel()
        unordered = [1, 3, 2]
        ordered = [1, 2, 3]
        m.I = FiniteSetOf(unordered)
        m.x = Var(m.I)
        unordered = [(i, m.x[i]) for i in unordered]
        ordered = [(i, m.x[i]) for i in ordered]
        self.assertEqual(list(m.x.items()), unordered)
        self.assertEqual(list(m.x.items(SortComponents.ORDERED_INDICES)), ordered)
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.items(True)), ordered)
        self.assertEqual(LOG.getvalue(), '')
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.items(ordered=True)), ordered)
        self.assertIn('items(ordered=True) is deprecated', LOG.getvalue())
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.items(ordered=False)), unordered)
        self.assertIn('items(ordered=False) is deprecated', LOG.getvalue())
        m = ConcreteModel()
        unordered = [1, 3, 2]
        ordered = [1, 2, 3]
        m.I = OrderedSetOf(unordered)
        m.x = Var(m.I)
        unordered = [(i, m.x[i]) for i in unordered]
        ordered = [(i, m.x[i]) for i in ordered]
        self.assertEqual(list(m.x.items()), unordered)
        self.assertEqual(list(m.x.items(SortComponents.ORDERED_INDICES)), unordered)
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.items(True)), ordered)
        self.assertEqual(LOG.getvalue(), '')
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.items(ordered=True)), unordered)
        self.assertIn('items(ordered=True) is deprecated', LOG.getvalue())
        with LoggingIntercept() as LOG:
            self.assertEqual(list(m.x.items(ordered=False)), unordered)
        self.assertIn('items(ordered=False) is deprecated', LOG.getvalue())

    def test_index_attribute_out_of_sync(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3])
        for i in [1, 2, 3]:
            self.assertEqual(m.x[i].index(), i)
        m.x[3]._index = 2
        with self.assertRaisesRegex(DeveloperError, ".*The '_data' dictionary and '_index' attribute are out of sync for indexed Var 'x': The 2 entry in the '_data' dictionary does not map back to this component data object.", normalize_whitespace=True):
            m.x[3].index()