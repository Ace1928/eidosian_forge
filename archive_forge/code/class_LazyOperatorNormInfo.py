from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """

    def __init__(self, A, A_1_norm=None, ell=2, scale=1):
        """
        Provide the operator and some norm-related information.

        Parameters
        ----------
        A : linear operator
            The operator of interest.
        A_1_norm : float, optional
            The exact 1-norm of A.
        ell : int, optional
            A technical parameter controlling norm estimation quality.
        scale : int, optional
            If specified, return the norms of scale*A instead of A.

        """
        self._A = A
        self._A_1_norm = A_1_norm
        self._ell = ell
        self._d = {}
        self._scale = scale

    def set_scale(self, scale):
        """
        Set the scale parameter.
        """
        self._scale = scale

    def onenorm(self):
        """
        Compute the exact 1-norm.
        """
        if self._A_1_norm is None:
            self._A_1_norm = _exact_1_norm(self._A)
        return self._scale * self._A_1_norm

    def d(self, p):
        """
        Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.
        """
        if p not in self._d:
            est = _onenormest_matrix_power(self._A, p, self._ell)
            self._d[p] = est ** (1.0 / p)
        return self._scale * self._d[p]

    def alpha(self, p):
        """
        Lazily compute max(d(p), d(p+1)).
        """
        return max(self.d(p), self.d(p + 1))