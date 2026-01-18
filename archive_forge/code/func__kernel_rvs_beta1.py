import numpy as np
from scipy import stats
from statsmodels.tools.rng_qrng import check_random_state
from statsmodels.distributions.copula.copulas import Copula
def _kernel_rvs_beta1(x, bw):
    return stats.beta.rvs(x / bw, (1 - x) / bw + 1)