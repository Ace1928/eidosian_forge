from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet
import logging
def _any_common_elements(a, b):
    if len(a) < len(b):
        for i in a:
            if i in b:
                return True
    else:
        for i in b:
            if i in a:
                return True
    return False