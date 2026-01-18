import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
def _rvs_scalar(self, chi, numsamples=None, random_state=None):
    size1d = tuple(np.atleast_1d(numsamples))
    N = int(np.prod(size1d))
    x = np.zeros(N)
    simulated = 0
    chi2 = chi * chi
    if chi <= 0.5:
        d = -chi2 / 2
        while simulated < N:
            k = N - simulated
            u = random_state.uniform(size=k)
            v = random_state.uniform(size=k)
            z = v ** (2 / 3)
            accept = np.log(u) <= d * z
            num_accept = np.sum(accept)
            if num_accept > 0:
                rvs = np.sqrt(1 - z[accept])
                x[simulated:simulated + num_accept] = rvs
                simulated += num_accept
    elif chi <= 1.8:
        echi = np.exp(-chi2 / 2)
        while simulated < N:
            k = N - simulated
            u = random_state.uniform(size=k)
            v = random_state.uniform(size=k)
            z = 2 * np.log(echi * (1 - v) + v) / chi2
            accept = u ** 2 + z <= 0
            num_accept = np.sum(accept)
            if num_accept > 0:
                rvs = np.sqrt(1 + z[accept])
                x[simulated:simulated + num_accept] = rvs
                simulated += num_accept
    else:
        while simulated < N:
            k = N - simulated
            g = random_state.standard_gamma(1.5, size=k)
            accept = g <= chi2 / 2
            num_accept = np.sum(accept)
            if num_accept > 0:
                x[simulated:simulated + num_accept] = g[accept]
                simulated += num_accept
        x = np.sqrt(1 - 2 * x / chi2)
    return np.reshape(x, size1d)