import warnings
import numpy as np
from scipy.sparse import issparse
from scipy.sparse._sputils import isshape, isintlike, asmatrix, is_pydata_spmatrix
def _rdot(self, x):
    """Matrix-matrix or matrix-vector multiplication from the right.

        Parameters
        ----------
        x : array_like
            1-d or 2-d array, representing a vector or matrix.

        Returns
        -------
        xA : array
            1-d or 2-d array (depending on the shape of x) that represents
            the result of applying this linear operator on x from the right.

        Notes
        -----
        This is copied from dot to implement right multiplication.
        """
    if isinstance(x, LinearOperator):
        return _ProductLinearOperator(x, self)
    elif np.isscalar(x):
        return _ScaledLinearOperator(self, x)
    else:
        if not issparse(x) and (not is_pydata_spmatrix(x)):
            x = np.asarray(x)
        if x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 1):
            return self.T.matvec(x.T).T
        elif x.ndim == 2:
            return self.T.matmat(x.T).T
        else:
            raise ValueError('expected 1-d or 2-d array or matrix, got %r' % x)