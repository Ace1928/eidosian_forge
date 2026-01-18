import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
class TestSliceComponent(_TestFlattenBase, unittest.TestCase):

    def make_model(self):
        m = ConcreteModel()
        m.s1 = Set(initialize=[1, 2, 3])
        m.s2 = Set(initialize=['a', 'b'])
        m.s3 = Set(initialize=[4, 5, 6])
        m.s4 = Set(initialize=['c', 'd'])
        m.v12 = Var(m.s1, m.s2)
        m.v124 = Var(m.s1, m.s2, m.s4)
        return m

    def test_no_sets(self):
        m = self.make_model()
        var = m.v12
        sets = (m.s3, m.s4)
        ref_data = {self._hashRef(v) for v in m.v12.values()}
        slices = [slice_ for _, slice_ in slice_component_along_sets(var, sets)]
        self.assertEqual(len(slices), len(ref_data))
        self.assertEqual(len(slices), len(m.s1) * len(m.s2))
        for slice_ in slices:
            self.assertIn(self._hashRef(slice_), ref_data)

    def test_one_set(self):
        m = self.make_model()
        var = m.v124
        sets = (m.s1, m.s3)
        ref_data = {self._hashRef(Reference(m.v124[:, i, j])) for i, j in m.s2 * m.s4}
        slices = [s for _, s in slice_component_along_sets(var, sets)]
        self.assertEqual(len(slices), len(ref_data))
        self.assertEqual(len(slices), len(m.s2) * len(m.s4))
        for slice_ in slices:
            self.assertIn(self._hashRef(Reference(slice_)), ref_data)

    def test_some_sets(self):
        m = self.make_model()
        var = m.v124
        sets = (m.s1, m.s3)
        ref_data = {self._hashRef(Reference(m.v124[:, i, j])) for i, j in m.s2 * m.s4}
        slices = [s for _, s in slice_component_along_sets(var, sets)]
        self.assertEqual(len(slices), len(ref_data))
        self.assertEqual(len(slices), len(m.s2) * len(m.s4))
        for slice_ in slices:
            self.assertIn(self._hashRef(Reference(slice_)), ref_data)

    def test_all_sets(self):
        m = self.make_model()
        var = m.v12
        sets = (m.s1, m.s2)
        ref_data = {self._hashRef(Reference(m.v12[:, :]))}
        slices = [s for _, s in slice_component_along_sets(var, sets)]
        self.assertEqual(len(slices), len(ref_data))
        self.assertEqual(len(slices), 1)
        for slice_ in slices:
            self.assertIn(self._hashRef(Reference(slice_)), ref_data)