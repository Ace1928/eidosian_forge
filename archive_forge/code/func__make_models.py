import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.mpc.interfaces.var_linker import DynamicVarLinker
def _make_models(self, n_time_points_1=3, n_time_points_2=3):
    m1 = pyo.ConcreteModel()
    m1.time = pyo.Set(initialize=range(n_time_points_1))
    m1.comp = pyo.Set(initialize=['A', 'B'])
    m1.var = pyo.Var(m1.time, m1.comp, initialize={(i, j): 1.0 + i * 0.1 for i, j in m1.time * m1.comp})
    m1.input = pyo.Var(m1.time, initialize={i: 1.0 - i * 0.1 for i in m1.time})
    m2 = pyo.ConcreteModel()
    m2.time = pyo.Set(initialize=range(n_time_points_2))
    m2.x1 = pyo.Var(m2.time, initialize=2.1)
    m2.x2 = pyo.Var(m2.time, initialize=2.2)
    m2.x3 = pyo.Var(m2.time, initialize=2.3)
    m2.x4 = pyo.Var(m2.time, initialize=2.4)
    return (m1, m2)