import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
def _get_infeasible_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(within=pyo.Binary)
    m.y = pyo.Var(within=pyo.NonNegativeReals)
    m.c1 = pyo.Constraint(expr=m.y <= 100.0 * m.x)
    m.c2 = pyo.Constraint(expr=m.y <= -100.0 * m.x)
    m.c3 = pyo.Constraint(expr=m.x >= 0.5)
    m.o = pyo.Objective(expr=-m.y)
    return m