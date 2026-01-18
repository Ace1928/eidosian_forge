import pyomo.common.unittest as unittest
import os
from pyomo.contrib.pynumero.dependencies import (
from pyomo.contrib.pynumero.asl import AmplInterface
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import (
def _make_model_with_inequalities(self):
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=range(4))
    m.x = pyo.Var(m.I, initialize=1.1)
    m.obj = pyo.Objective(expr=1 * m.x[0] + 2 * m.x[1] ** 2 + 3 * m.x[1] * m.x[2] + 4 * m.x[3] ** 3)
    m.eq_con_1 = pyo.Constraint(expr=m.x[0] * m.x[1] ** 1.1 * m.x[2] ** 1.2 == 3.0)
    m.eq_con_2 = pyo.Constraint(expr=m.x[0] ** 2 + m.x[3] ** 2 + m.x[1] == 2.0)
    m.ineq_con_1 = pyo.Constraint(expr=m.x[0] + m.x[3] * m.x[0] <= 4.0)
    m.ineq_con_2 = pyo.Constraint(expr=m.x[1] + m.x[2] >= 1.0)
    m.ineq_con_3 = pyo.Constraint(expr=m.x[2] >= 0)
    return m