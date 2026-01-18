import pyomo.environ as pyo
import math
from pyomo.core.base.block import _BlockData
from pyomo.common.collections import ComponentSet
from pyomo.core.base.var import _GeneralVarData
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
import logging
def _bounds_to_float(lb, ub):
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf
    return (lb, ub)