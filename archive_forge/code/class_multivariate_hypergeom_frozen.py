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
class multivariate_hypergeom_frozen(multi_rv_frozen):

    def __init__(self, m, n, seed=None):
        self._dist = multivariate_hypergeom_gen(seed)
        self.M, self.m, self.n, self.mcond, self.ncond, self.mncond = self._dist._process_parameters(m, n)

        def _process_parameters(m, n):
            return (self.M, self.m, self.n, self.mcond, self.ncond, self.mncond)
        self._dist._process_parameters = _process_parameters

    def logpmf(self, x):
        return self._dist.logpmf(x, self.m, self.n)

    def pmf(self, x):
        return self._dist.pmf(x, self.m, self.n)

    def mean(self):
        return self._dist.mean(self.m, self.n)

    def var(self):
        return self._dist.var(self.m, self.n)

    def cov(self):
        return self._dist.cov(self.m, self.n)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.m, self.n, size=size, random_state=random_state)