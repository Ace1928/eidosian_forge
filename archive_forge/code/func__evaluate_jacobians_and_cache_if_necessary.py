from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
def _evaluate_jacobians_and_cache_if_necessary(self):
    if not self._jac_full_is_cached:
        self._asl.eval_jac_g(self._primals, self._cached_jac_full.data)
        self._jac_full_is_cached = True