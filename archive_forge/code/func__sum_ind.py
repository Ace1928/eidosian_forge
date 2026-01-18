import numpy as np
from scipy import sparse
from pygsp import utils
def _sum_ind(ind1, ind2):
    ind = np.tile(np.ravel(ind1), (np.size(ind2), 1)).T + np.ravel(ind2)
    return np.ravel(ind)