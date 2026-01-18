import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def _model1_1d_sets(self):
    m = ConcreteModel()
    m.time = Set(initialize=[1, 2, 3])
    m.space = Set(initialize=[0.0, 0.5, 1.0])
    m.comp = Set(initialize=['a', 'b'])
    m.v0 = Var()
    m.v1 = Var(m.time)
    m.v2 = Var(m.time, m.space)
    m.v3 = Var(m.time, m.space, m.comp)
    m.v_tt = Var(m.time, m.time)
    m.v_tst = Var(m.time, m.space, m.time)

    @m.Block()
    def b(b):

        @b.Block(m.time)
        def b1(b1):
            b1.v0 = Var()
            b1.v1 = Var(m.space)
            b1.v2 = Var(m.space, m.comp)

            @b1.Block(m.space)
            def b_s(b_s):
                b_s.v0 = Var()
                b_s.v1 = Var(m.space)
                b_s.v2 = Var(m.space, m.comp)

        @b.Block(m.time, m.space)
        def b2(b2):
            b2.v0 = Var()
            b2.v1 = Var(m.comp)
            b2.v2 = Var(m.time, m.comp)
    return m