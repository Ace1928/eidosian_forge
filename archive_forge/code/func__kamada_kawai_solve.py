import networkx as nx
from networkx.utils import np_random_state
def _kamada_kawai_solve(dist_mtx, pos_arr, dim):
    import numpy as np
    import scipy as sp
    meanwt = 0.001
    costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 0.001), meanwt, dim)
    optresult = sp.optimize.minimize(_kamada_kawai_costfn, pos_arr.ravel(), method='L-BFGS-B', args=costargs, jac=True)
    return optresult.x.reshape((-1, dim))