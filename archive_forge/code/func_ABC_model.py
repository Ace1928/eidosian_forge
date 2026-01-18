from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
def ABC_model(data):
    ca_meas = data['ca']
    cb_meas = data['cb']
    cc_meas = data['cc']
    if isinstance(data, pd.DataFrame):
        meas_t = data.index
    else:
        meas_t = list(ca_meas.keys())
    ca0 = 1.0
    cb0 = 0.0
    cc0 = 0.0
    m = pyo.ConcreteModel()
    m.k1 = pyo.Var(initialize=0.5, bounds=(0.0001, 10))
    m.k2 = pyo.Var(initialize=3.0, bounds=(0.0001, 10))
    m.time = dae.ContinuousSet(bounds=(0.0, 5.0), initialize=meas_t)
    m.ca = pyo.Var(m.time, initialize=ca0, bounds=(-0.001, ca0 + 0.001))
    m.cb = pyo.Var(m.time, initialize=cb0, bounds=(-0.001, ca0 + 0.001))
    m.cc = pyo.Var(m.time, initialize=cc0, bounds=(-0.001, ca0 + 0.001))
    m.dca = dae.DerivativeVar(m.ca, wrt=m.time)
    m.dcb = dae.DerivativeVar(m.cb, wrt=m.time)
    m.dcc = dae.DerivativeVar(m.cc, wrt=m.time)

    def _dcarate(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        else:
            return m.dca[t] == -m.k1 * m.ca[t]
    m.dcarate = pyo.Constraint(m.time, rule=_dcarate)

    def _dcbrate(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        else:
            return m.dcb[t] == m.k1 * m.ca[t] - m.k2 * m.cb[t]
    m.dcbrate = pyo.Constraint(m.time, rule=_dcbrate)

    def _dccrate(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        else:
            return m.dcc[t] == m.k2 * m.cb[t]
    m.dccrate = pyo.Constraint(m.time, rule=_dccrate)

    def ComputeFirstStageCost_rule(m):
        return 0
    m.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(m):
        return sum(((m.ca[t] - ca_meas[t]) ** 2 + (m.cb[t] - cb_meas[t]) ** 2 + (m.cc[t] - cc_meas[t]) ** 2 for t in meas_t))
    m.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    def total_cost_rule(model):
        return model.FirstStageCost + model.SecondStageCost
    m.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)
    disc = pyo.TransformationFactory('dae.collocation')
    disc.apply_to(m, nfe=20, ncp=2)
    return m