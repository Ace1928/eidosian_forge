import numpy as np
from numpy import array, asarray, float64, zeros
from . import _lbfgsb
from ._optimize import (MemoizeJac, OptimizeResult, _call_callback_maybe_halt,
from ._constraints import old_bound_to_new
from scipy.sparse.linalg import LinearOperator
class LbfgsInvHessProduct(LinearOperator):
    """Linear operator for the L-BFGS approximate inverse Hessian.

    This operator computes the product of a vector with the approximate inverse
    of the Hessian of the objective function, using the L-BFGS limited
    memory approximation to the inverse Hessian, accumulated during the
    optimization.

    Objects of this class implement the ``scipy.sparse.linalg.LinearOperator``
    interface.

    Parameters
    ----------
    sk : array_like, shape=(n_corr, n)
        Array of `n_corr` most recent updates to the solution vector.
        (See [1]).
    yk : array_like, shape=(n_corr, n)
        Array of `n_corr` most recent updates to the gradient. (See [1]).

    References
    ----------
    .. [1] Nocedal, Jorge. "Updating quasi-Newton matrices with limited
       storage." Mathematics of computation 35.151 (1980): 773-782.

    """

    def __init__(self, sk, yk):
        """Construct the operator."""
        if sk.shape != yk.shape or sk.ndim != 2:
            raise ValueError('sk and yk must have matching shape, (n_corrs, n)')
        n_corrs, n = sk.shape
        super().__init__(dtype=np.float64, shape=(n, n))
        self.sk = sk
        self.yk = yk
        self.n_corrs = n_corrs
        self.rho = 1 / np.einsum('ij,ij->i', sk, yk)

    def _matvec(self, x):
        """Efficient matrix-vector multiply with the BFGS matrices.

        This calculation is described in Section (4) of [1].

        Parameters
        ----------
        x : ndarray
            An array with shape (n,) or (n,1).

        Returns
        -------
        y : ndarray
            The matrix-vector product

        """
        s, y, n_corrs, rho = (self.sk, self.yk, self.n_corrs, self.rho)
        q = np.array(x, dtype=self.dtype, copy=True)
        if q.ndim == 2 and q.shape[1] == 1:
            q = q.reshape(-1)
        alpha = np.empty(n_corrs)
        for i in range(n_corrs - 1, -1, -1):
            alpha[i] = rho[i] * np.dot(s[i], q)
            q = q - alpha[i] * y[i]
        r = q
        for i in range(n_corrs):
            beta = rho[i] * np.dot(y[i], r)
            r = r + s[i] * (alpha[i] - beta)
        return r

    def todense(self):
        """Return a dense array representation of this operator.

        Returns
        -------
        arr : ndarray, shape=(n, n)
            An array with the same shape and containing
            the same data represented by this `LinearOperator`.

        """
        s, y, n_corrs, rho = (self.sk, self.yk, self.n_corrs, self.rho)
        I = np.eye(*self.shape, dtype=self.dtype)
        Hk = I
        for i in range(n_corrs):
            A1 = I - s[i][:, np.newaxis] * y[i][np.newaxis, :] * rho[i]
            A2 = I - y[i][:, np.newaxis] * s[i][np.newaxis, :] * rho[i]
            Hk = np.dot(A1, np.dot(Hk, A2)) + rho[i] * s[i][:, np.newaxis] * s[i][np.newaxis, :]
        return Hk