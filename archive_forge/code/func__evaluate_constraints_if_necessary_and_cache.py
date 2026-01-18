import os
import numpy as np
import logging
from scipy.sparse import coo_matrix, identity
from pyomo.common.deprecation import deprecated
import pyomo.core.base as pyo
from pyomo.common.collections import ComponentMap
from pyomo.contrib.pynumero.sparse.block_matrix import BlockMatrix
from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.utils import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedNLP
def _evaluate_constraints_if_necessary_and_cache(self):
    if self._cached_constraint_residuals is None:
        c = BlockVector(2)
        if self._ex_model.n_equality_constraints() > 0:
            c.set_block(0, self._ex_model.evaluate_equality_constraints())
        else:
            c.set_block(0, np.zeros(0, dtype=np.float64))
        if self._ex_model.n_outputs() > 0:
            output_values = self._primal_values[self._ex_model.n_inputs():]
            c.set_block(1, self._ex_model.evaluate_outputs() - output_values)
        else:
            c.set_block(1, np.zeros(0, dtype=np.float64))
        self._cached_constraint_residuals = c.flatten()