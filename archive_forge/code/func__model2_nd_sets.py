import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def _model2_nd_sets(self):
    m = ConcreteModel()
    normalize_index.flatten = False
    m.d1 = Set(initialize=[1, 2])
    m.d2 = Set(initialize=[('a', 1), ('b', 2)])
    m.dn = Set(initialize=[('c', 3), ('d', 4, 5)], dimen=None)
    m.v_2n = Var(m.d2, m.dn)
    m.v_12 = Var(m.d1, m.d2)
    m.v_212 = Var(m.d2, m.d1, m.d2)
    m.v_12n = Var(m.d1, m.d2, m.dn)
    m.v_1n2n = Var(m.d1, m.dn, m.d2, m.dn)

    @m.Block(m.d1, m.d2, m.dn)
    def b(b, i1, i2, i3):
        b.v0 = Var()
        b.v1 = Var(m.d1)
        b.v2 = Var(m.d2)
        b.vn = Var(m.dn)
    normalize_index.flatten = True
    return m