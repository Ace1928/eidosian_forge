import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state
def _ppf(self, q):
    """Percent point function (inverse of `cdf`)

        Parameters
        ----------
        q : array_like
            lower tail probability

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.

        """
    if self._p_domain == 1.0:
        return self._frozendist.ppf(q)
    x = self._frozendist.ppf(self._p_domain * np.array(q) + self._p_lower)
    return np.clip(x, self._domain_adj[0], self._domain_adj[1])