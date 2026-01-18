import warnings
import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import factorial
from scipy.stats import rv_continuous
import scipy.special as special
def _compute_coefs_pdf(self, cum):
    mu, sigma = (cum[0], np.sqrt(cum[1]))
    lam = np.asarray(cum)
    for j, l in enumerate(lam):
        lam[j] /= cum[1] ** j
    coef = np.zeros(lam.size * 3 - 5)
    coef[0] = 1.0
    for s in range(lam.size - 2):
        for p in _faa_di_bruno_partitions(s + 1):
            term = sigma ** (s + 1)
            for m, k in p:
                term *= np.power(lam[m + 1] / factorial(m + 2), k) / factorial(k)
            r = sum((k for m, k in p))
            coef[s + 1 + 2 * r] += term
    return (coef, mu, sigma)