import warnings
import numpy as np
from numpy.linalg import eigh, inv, norm, matrix_rank
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.graphics.utils import _import_mpl
from .factor_rotation import rotate_factors, promax
def _fit_ml_em(self, iter, random_state=None):
    """estimate Factor model using EM algorithm
        """
    if random_state is None:
        random_state = np.random.RandomState(3427)
    load = 0.1 * random_state.standard_normal(size=(self.k_endog, self.n_factor))
    uniq = 0.5 * np.ones(self.k_endog)
    for k in range(iter):
        loadu = load / uniq[:, None]
        f = np.dot(load.T, loadu)
        f.flat[::f.shape[0] + 1] += 1
        r = np.linalg.solve(f, loadu.T)
        q = np.dot(loadu.T, load)
        h = np.dot(r, load)
        c = load - np.dot(load, h)
        c /= uniq[:, None]
        g = np.dot(q, r)
        e = np.dot(g, self.corr)
        d = np.dot(loadu.T, self.corr) - e
        a = np.dot(d, c)
        a -= np.dot(load.T, c)
        a.flat[::a.shape[0] + 1] += 1
        b = np.dot(self.corr, c)
        load = np.linalg.solve(a, b.T).T
        uniq = np.diag(self.corr) - (load * d.T).sum(1)
    return (load, uniq)