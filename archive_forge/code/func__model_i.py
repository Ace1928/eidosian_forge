import sys
import pyomo.environ as pyo
import numpy.random as rnd
from pyomo.common.dependencies import pandas as pd
import pyomo.contrib.pynumero.examples.external_grey_box.param_est.models as po
def _model_i(b, i):
    po.build_single_point_model_external(b)