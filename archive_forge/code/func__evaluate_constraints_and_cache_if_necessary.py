from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def _evaluate_constraints_and_cache_if_necessary(self):
    if not self._con_full_is_cached:
        self._asl.eval_g(self._primals, self._cached_con_full)
        self._cached_con_full -= self._con_full_rhs
        self._con_full_is_cached = True