import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _compute_using_nipals(self):
    """
        NIPALS implementation to compute small number of eigenvalues
        and eigenvectors
        """
    x = self.transformed_data
    if self._ncomp > 1:
        x = x + 0.0
    tol, max_iter, ncomp = (self._tol, self._max_iter, self._ncomp)
    vals = np.zeros(self._ncomp)
    vecs = np.zeros((self._nvar, self._ncomp))
    for i in range(ncomp):
        max_var_ind = np.argmax(x.var(0))
        factor = x[:, [max_var_ind]]
        _iter = 0
        diff = 1.0
        while diff > tol and _iter < max_iter:
            vec = x.T.dot(factor) / factor.T.dot(factor)
            vec = vec / np.sqrt(vec.T.dot(vec))
            factor_last = factor
            factor = x.dot(vec) / vec.T.dot(vec)
            diff = _norm(factor - factor_last) / _norm(factor)
            _iter += 1
        vals[i] = (factor ** 2).sum()
        vecs[:, [i]] = vec
        if ncomp > 1:
            x -= factor.dot(vec.T)
    self.eigenvals = vals
    self.eigenvecs = vecs