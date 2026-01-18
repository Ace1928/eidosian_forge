from pyomo.contrib.pynumero.interfaces.nlp import NLP, ExtendedNLP
import numpy as np
import scipy.sparse as sp
def _project_primals(self, default, original_primals):
    projected_x = default * np.ones(self.n_primals(), dtype=np.float64)
    projected_x[self._projected_idxs] = original_primals[self._original_idxs]
    return projected_x