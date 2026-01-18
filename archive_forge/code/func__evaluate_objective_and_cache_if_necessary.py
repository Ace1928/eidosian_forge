from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def _evaluate_objective_and_cache_if_necessary(self):
    if not self._objective_is_cached:
        self._cached_objective = self._asl.eval_f(self._primals)
        self._objective_is_cached = True