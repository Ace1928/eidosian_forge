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
class unitary_group_frozen(multi_rv_frozen):

    def __init__(self, dim=None, seed=None):
        """Create a frozen (U(N)) n-dimensional unitary matrix distribution.

        Parameters
        ----------
        dim : scalar
            Dimension of matrices
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Examples
        --------
        >>> from scipy.stats import unitary_group
        >>> x = unitary_group(3)
        >>> x.rvs()

        """
        self._dist = unitary_group_gen(seed)
        self.dim = self._dist._process_parameters(dim)

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.dim, size, random_state)