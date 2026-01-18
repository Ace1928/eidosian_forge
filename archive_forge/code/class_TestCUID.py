import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
class TestCUID(unittest.TestCase):

    def test_cuid_of_slice(self):
        m = pyo.ConcreteModel()
        m.s1 = pyo.Set(initialize=['a', 'b'])
        m.s2 = pyo.Set(initialize=['c', 'd'])
        m.b = pyo.Block(m.s1)
        for i in m.s1:
            m.b[i].v = pyo.Var(m.s2)
        slice_ = slice_component_along_sets(m.b['a'].v['c'], ComponentSet((m.s1,)))
        cuid = pyo.ComponentUID(slice_)
        self.assertEqual(str(cuid), 'b[*].v[c]')
        slice_ = slice_component_along_sets(m.b['a'].v['c',], ComponentSet((m.s1,)))
        cuid = pyo.ComponentUID(slice_)
        self.assertEqual(str(cuid), 'b[*].v[c]')