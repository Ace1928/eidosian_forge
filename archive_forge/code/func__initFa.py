import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def _initFa(m, t):
    if t < 10800:
        return data['Fa1']
    else:
        return data['Fa2']