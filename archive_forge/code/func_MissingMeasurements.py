import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def MissingMeasurements(m):
    if data['experiment'] == 1:
        return sum(((m.Ca[t] - m.Ca_meas[t]) ** 2 + (m.Cb[t] - m.Cb_meas[t]) ** 2 + (m.Cc[t] - m.Cc_meas[t]) ** 2 + (m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT))
    elif data['experiment'] == 2:
        return sum(((m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT))
    else:
        return sum(((m.Cb[t] - m.Cb_meas[t]) ** 2 + (m.Tr[t] - m.Tr_meas[t]) ** 2 for t in m.measT))