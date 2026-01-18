import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def _initTc(m, t):
    if t < 10800:
        return data['Tc1']
    else:
        return data['Tc2']