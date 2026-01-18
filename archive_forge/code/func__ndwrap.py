import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.contrib.interior_point.inverse_reduced_hessian import (
def _ndwrap(x):
    model.b0.fix(x[0])
    model.b['tofu'].fix(x[1])
    model.b['chard'].fix(x[2])
    rval = pyo.value(model.SSE)
    return rval