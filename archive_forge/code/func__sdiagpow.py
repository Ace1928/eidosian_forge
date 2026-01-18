import numpy as np
from scipy import linalg
from statsmodels.tools.decorators import cache_readonly
def _sdiagpow(self, p):
    return linalg.diagsvd(np.power(self.s, p), *x.shape)