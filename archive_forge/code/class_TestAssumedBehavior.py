import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
class TestAssumedBehavior(unittest.TestCase):
    """
    These are some behaviors we rely on that weren't
    immediately obvious would be the case.
    """

    def setUp(self):
        self._orig_flatten = normalize_index.flatten

    def tearDown(self):
        normalize_index.flatten = self._orig_flatten

    def test_cross(self):
        m = ConcreteModel()
        m.s1 = Set(initialize=[1, 2])
        m.s2 = Set(initialize=[3, 4])
        m.s3 = Set(initialize=['a', 'b'])
        normalize_index.flatten = True
        for i in m.s1.cross():
            self.assertIs(type(i), tuple)
        for i in m.s1.cross(m.s2, m.s3):
            self.assertIs(type(i), tuple)
            for j in i:
                self.assertIsNot(type(j), tuple)
        normalize_index.flatten = False
        for i in m.s1.cross():
            self.assertIs(type(i), tuple)
        for i in m.s1.cross(m.s2, m.s3):
            self.assertIs(type(i), tuple)
            for j in i:
                self.assertIsNot(type(j), tuple)
        normalize_index.flatten = True

    def test_subsets(self):
        m = ConcreteModel()
        m.s1 = Set(initialize=[1, 2])
        m.s2 = Set(initialize=[3, 4])
        m.s3 = Set(initialize=['a', 'b'])
        normalize_index.flatten = True
        s12 = m.s1 * m.s2
        s12_3 = s12 * m.s3
        s123 = m.s1.cross(m.s2, m.s3)
        subsets12_3 = list(s12_3.subsets())
        subsets123 = list(s123.subsets())
        self.assertEqual(len(subsets12_3), len(subsets123))
        for s_a, s_b in zip(subsets12_3, subsets123):
            self.assertIs(s_a, s_b)
        normalize_index.flatten = False
        for i, j in s12_3:
            self.assertIs(type(i), tuple)
        subsets12_3 = list(s12_3.subsets())
        subsets123 = list(s123.subsets())
        self.assertEqual(len(subsets12_3), len(subsets123))
        for s_a, s_b in zip(subsets12_3, subsets123):
            self.assertIs(s_a, s_b)
        normalize_index.flatten = True