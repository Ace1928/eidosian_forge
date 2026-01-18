import numpy as np
import abc
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.environ import Var, Constraint, value
from pyomo.core.base.var import _VarData
from pyomo.common.modeling import unique_component_name
def _ex_io_inputs_from_full_primals(self, primals):
    return primals[self._input_columns]