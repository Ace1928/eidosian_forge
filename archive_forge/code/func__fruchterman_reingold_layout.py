from collections import Counter
import numpy as np
from scipy import sparse
from pygsp import utils
from . import fourier, difference  # prevent circular import in Python < 3.5
def _fruchterman_reingold_layout(self, dim=2, k=None, pos=None, fixed=[], iterations=50, scale=1.0, center=None, seed=None):
    if center is None:
        center = np.zeros((1, dim))
    if np.shape(center)[1] != dim:
        self.logger.error('Spring coordinates: center has wrong size.')
        center = np.zeros((1, dim))
    if pos is None:
        dom_size = 1
        pos_arr = None
    else:
        dom_size = np.max(pos)
        pos_arr = np.random.RandomState(seed).uniform(size=(self.N, dim))
        pos_arr = pos_arr * dom_size + center
        for i in range(self.N):
            pos_arr[i] = np.asarray(pos[i])
    if k is None and len(fixed) > 0:
        k = dom_size / np.sqrt(self.N)
    pos = _sparse_fruchterman_reingold(self.A, dim, k, pos_arr, fixed, iterations, seed)
    if len(fixed) == 0:
        pos = _rescale_layout(pos, scale=scale) + center
    return pos