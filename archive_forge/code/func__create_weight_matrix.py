import numpy as np
from scipy import sparse
from pygsp import utils
from . import Graph  # prevent circular import in Python < 3.5
def _create_weight_matrix(self, N, param_distribute, regular, param_Nc):
    XCoords = np.zeros((N, 1))
    YCoords = np.zeros((N, 1))
    rs = np.random.RandomState(self.seed)
    if param_distribute:
        mdim = int(np.ceil(np.sqrt(N)))
        for i in range(mdim):
            for j in range(mdim):
                if i * mdim + j < N:
                    XCoords[i * mdim + j] = np.array((i + rs.rand()) / mdim)
                    YCoords[i * mdim + j] = np.array((j + rs.rand()) / mdim)
    else:
        XCoords = rs.rand(N, 1)
        YCoords = rs.rand(N, 1)
    coords = np.concatenate((XCoords, YCoords), axis=1)
    target_dist_cutoff = 2 * N ** (-0.5)
    T = 0.6
    s = np.sqrt(-target_dist_cutoff ** 2 / (2 * np.log(T)))
    d = utils.distanz(x=coords.T)
    W = np.exp(-d ** 2 / (2.0 * s ** 2))
    W -= np.diag(np.diag(W))
    if regular:
        W = self._get_nc_connection(W, param_Nc)
    else:
        W2 = self._get_nc_connection(W, param_Nc)
        W = np.where(W < T, 0, W)
        W = np.where(W2 > 0, W2, W)
    W = sparse.csc_matrix(W)
    return (W, coords)