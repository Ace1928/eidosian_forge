import numpy as np
from scipy import sparse
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
def _get_nc_connection(self, W, param_nc):
    Wtmp = W
    W = np.zeros(np.shape(W))
    for i in range(np.shape(W)[0]):
        l = Wtmp[i]
        for j in range(param_nc):
            val = np.max(l)
            ind = np.argmax(l)
            W[i, ind] = val
            l[ind] = 0
    W = utils.symmetrize(W, method='average')
    return W