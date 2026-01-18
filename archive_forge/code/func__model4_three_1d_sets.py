import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def _model4_three_1d_sets(self):
    m = ConcreteModel()
    m.X = Set(initialize=[1, 2, 3])
    m.Y = Set(initialize=[1, 2, 3])
    m.Z = Set(initialize=[1, 2, 3])
    m.comp = Set(initialize=['a', 'b'])
    m.u = Var()
    m.v = Var(m.X, m.Y, m.Z, m.comp)
    m.base = Var(m.X, m.Y)

    @m.Block(m.X, m.Y, m.Z, m.comp)
    def b4(b, x, y, z, j):
        b.v = Var()

    @m.Block(m.X, m.Y)
    def b2(b, x, y):
        b.base = Var()
        b.v = Var(m.Z, m.comp)
    return m