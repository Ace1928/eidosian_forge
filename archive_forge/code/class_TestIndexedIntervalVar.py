import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
class TestIndexedIntervalVar(unittest.TestCase):

    def test_initialize_with_no_data(self):
        m = ConcreteModel()
        m.i = IntervalVar([1, 2])
        for j in [1, 2]:
            self.assertIsInstance(m.i[j].start_time, IntervalVarTimePoint)
            self.assertEqual(m.i[j].start_time.domain, Integers)
            self.assertIsNone(m.i[j].start_time.lower)
            self.assertIsNone(m.i[j].start_time.upper)
            self.assertIsInstance(m.i[j].end_time, IntervalVarTimePoint)
            self.assertEqual(m.i[j].end_time.domain, Integers)
            self.assertIsNone(m.i[j].end_time.lower)
            self.assertIsNone(m.i[j].end_time.upper)
            self.assertIsInstance(m.i[j].length, IntervalVarLength)
            self.assertEqual(m.i[j].length.domain, Integers)
            self.assertIsNone(m.i[j].length.lower)
            self.assertIsNone(m.i[j].length.upper)
            self.assertIsInstance(m.i[j].is_present, IntervalVarPresence)

    def test_constant_length(self):
        m = ConcreteModel()
        m.i = IntervalVar(['a', 'b'], length=45)
        for j in ['a', 'b']:
            self.assertEqual(m.i[j].length.lower, 45)
            self.assertEqual(m.i[j].length.upper, 45)

    def test_rule_based_start(self):
        m = ConcreteModel()

        def start_rule(m, i):
            return (1 - i, 13 + i)
        m.act = IntervalVar([1, 2, 3], start=start_rule, length=4)
        for i in [1, 2, 3]:
            self.assertEqual(m.act[i].start_time.lower, 1 - i)
            self.assertEqual(m.act[i].start_time.upper, 13 + i)
            self.assertEqual(m.act[i].length.lower, 4)
            self.assertEqual(m.act[i].length.upper, 4)
            self.assertFalse(m.act[i].optional)
            self.assertTrue(m.act[i].is_present.fixed)
            self.assertEqual(value(m.act[i].is_present), True)

    def test_optional(self):
        m = ConcreteModel()
        m.act = IntervalVar([1, 2], end=[0, 10], optional=True)
        for i in [1, 2]:
            self.assertTrue(m.act[i].optional)
            self.assertFalse(m.act[i].is_present.fixed)
            self.assertEqual(m.act[i].end_time.lower, 0)
            self.assertEqual(m.act[i].end_time.upper, 10)
        with self.assertRaisesRegex(ValueError, "Cannot set 'optional' to None: Must be True or False."):
            m.act[1].optional = None
        m.act[1].optional = False
        self.assertFalse(m.act[1].optional)
        self.assertTrue(m.act[1].is_present.fixed)
        m.act[1].optional = True
        self.assertTrue(m.act[1].optional)
        self.assertFalse(m.act[1].is_present.fixed)

    def test_optional_rule(self):
        m = ConcreteModel()
        m.idx = Set(initialize=[(4, 2), (5, 2)], dimen=2)

        def optional_rule(m, i, j):
            return i % j == 0
        m.act = IntervalVar(m.idx, optional=optional_rule)
        self.assertTrue(m.act[4, 2].optional)
        self.assertFalse(m.act[5, 2].optional)

    def test_index_by_expr(self):
        m = ConcreteModel()
        m.act = IntervalVar([(1, 2), (2, 1), (2, 2)])
        m.i = Var(domain=Integers)
        m.i2 = Var([1, 2], domain=Integers)
        thing1 = m.act[m.i, 2]
        self.assertIsInstance(thing1, GetItemExpression)
        self.assertEqual(len(thing1.args), 3)
        self.assertIs(thing1.args[0], m.act)
        self.assertIs(thing1.args[1], m.i)
        self.assertEqual(thing1.args[2], 2)
        thing2 = thing1.start_time
        self.assertIsInstance(thing2, GetAttrExpression)
        self.assertEqual(len(thing2.args), 2)
        self.assertIs(thing2.args[0], thing1)
        self.assertEqual(thing2.args[1], 'start_time')
        expr1 = m.act[m.i, 2].start_time.before(m.act[m.i ** 2, 1].end_time)