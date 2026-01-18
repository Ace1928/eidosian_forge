import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
def _handle_u(self, u):
    u = np.asarray(u)
    if u.shape[-1] != self.k_dim:
        import warnings
        warnings.warn('u has different dimension than k_dim. This will raise exception in future versions', FutureWarning)
    return u