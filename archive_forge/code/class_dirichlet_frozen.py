import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
class dirichlet_frozen(multi_rv_frozen):

    def __init__(self, alpha, seed=None):
        self.alpha = _dirichlet_check_parameters(alpha)
        self._dist = dirichlet_gen(seed)

    def logpdf(self, x):
        return self._dist.logpdf(x, self.alpha)

    def pdf(self, x):
        return self._dist.pdf(x, self.alpha)

    def mean(self):
        return self._dist.mean(self.alpha)

    def var(self):
        return self._dist.var(self.alpha)

    def cov(self):
        return self._dist.cov(self.alpha)

    def entropy(self):
        return self._dist.entropy(self.alpha)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.alpha, size, random_state)