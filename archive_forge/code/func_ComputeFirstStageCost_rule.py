import json
from os.path import join, abspath, dirname
from pyomo.environ import (
from pyomo.dae import ContinuousSet, DerivativeVar
def ComputeFirstStageCost_rule(model):
    return 0