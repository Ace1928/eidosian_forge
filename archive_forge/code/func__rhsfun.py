import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _rhsfun(t, x):
    residual = []
    cstemplate.set_value(t)
    for idx, v in enumerate(diffvars):
        if v in templatemap:
            templatemap[v].set_value(x[idx])
    for d in derivlist:
        residual.append(rhsdict[d]())
    return residual