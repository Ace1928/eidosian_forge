import numpy as np
import numpy.linalg as la
import scipy.linalg as L
from statsmodels.tools.decorators import cache_readonly
import statsmodels.tsa.tsatools as tsa
import statsmodels.tsa.vector_ar.plotting as plotting
import statsmodels.tsa.vector_ar.util as util
def _make_g(i):
    G = 0.0
    for m in range(i):
        idx = i - 1 - m
        if idx in self._g_memo:
            apow = self._g_memo[idx]
        else:
            apow = la.matrix_power(self._A.T, idx)
            apow = apow[:K]
            self._g_memo[idx] = apow
        piece = np.kron(apow, self.irfs[m])
        G = G + piece
    return G