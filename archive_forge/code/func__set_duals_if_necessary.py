import numpy as np
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint, value
from pyomo.core.base.var import _VarData
from pyomo.common.modeling import unique_component_name
def _set_duals_if_necessary(self, duals):
    if not np.array_equal(duals, self._cached_duals):
        self._cached_duals.copy_from(duals)
        self._pyomo_nlp.set_duals(self._cached_duals.get_block(0))