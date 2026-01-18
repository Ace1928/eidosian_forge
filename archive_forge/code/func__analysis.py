import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def _analysis(g, s, **kwargs):
    s = g.filter(s, **kwargs)
    while s.ndim < 3:
        s = np.expand_dims(s, 1)
    return s.swapaxes(1, 2).reshape(-1, s.shape[1], order='F')