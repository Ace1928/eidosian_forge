import sys
import pyomo.environ as pyo
import numpy.random as rnd
from pyomo.common.dependencies import pandas as pd
import pyomo.contrib.pynumero.examples.external_grey_box.param_est.models as po
def _eq_parameter(m, i):
    return m.UA == m.model_i[i].egb.inputs['UA']