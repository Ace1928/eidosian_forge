import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def AllMeasurements(m):
    return sum(((m.Ca[t] - m.Ca_meas[t]) ** 2 + (m.Cb[t] - m.Cb_meas[t]) ** 2 + (m.Cc[t] - m.Cc_meas[t]) ** 2 + 0.01 * (m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT))