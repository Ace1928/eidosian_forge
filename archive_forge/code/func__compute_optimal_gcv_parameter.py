import operator
from math import prod
import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations
def _compute_optimal_gcv_parameter(X, wE, y, w):
    """
    Returns an optimal regularization parameter from the GCV criteria [1].

    Parameters
    ----------
    X : array, shape (5, n)
        5 bands of the design matrix ``X`` stored in LAPACK banded storage.
    wE : array, shape (5, n)
        5 bands of the penalty matrix :math:`W^{-1} E` stored in LAPACK banded
        storage.
    y : array, shape (n,)
        Ordinates.
    w : array, shape (n,)
        Vector of weights.

    Returns
    -------
    lam : float
        An optimal from the GCV criteria point of view regularization
        parameter.

    Notes
    -----
    No checks are performed.

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
        for observational data, Philadelphia, Pennsylvania: Society for
        Industrial and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`

    """

    def compute_banded_symmetric_XT_W_Y(X, w, Y):
        """
        Assuming that the product :math:`X^T W Y` is symmetric and both ``X``
        and ``Y`` are 5-banded, compute the unique bands of the product.

        Parameters
        ----------
        X : array, shape (5, n)
            5 bands of the matrix ``X`` stored in LAPACK banded storage.
        w : array, shape (n,)
            Array of weights
        Y : array, shape (5, n)
            5 bands of the matrix ``Y`` stored in LAPACK banded storage.

        Returns
        -------
        res : array, shape (4, n)
            The result of the product :math:`X^T Y` stored in the banded way.

        Notes
        -----
        As far as the matrices ``X`` and ``Y`` are 5-banded, their product
        :math:`X^T W Y` is 7-banded. It is also symmetric, so we can store only
        unique diagonals.

        """
        W_Y = np.copy(Y)
        W_Y[2] *= w
        for i in range(2):
            W_Y[i, 2 - i:] *= w[:-2 + i]
            W_Y[3 + i, :-1 - i] *= w[1 + i:]
        n = X.shape[1]
        res = np.zeros((4, n))
        for i in range(n):
            for j in range(min(n - i, 4)):
                res[-j - 1, i + j] = sum(X[j:, i] * W_Y[:5 - j, i + j])
        return res

    def compute_b_inv(A):
        """
        Inverse 3 central bands of matrix :math:`A=U^T D^{-1} U` assuming that
        ``U`` is a unit upper triangular banded matrix using an algorithm
        proposed in [1].

        Parameters
        ----------
        A : array, shape (4, n)
            Matrix to inverse, stored in LAPACK banded storage.

        Returns
        -------
        B : array, shape (4, n)
            3 unique bands of the symmetric matrix that is an inverse to ``A``.
            The first row is filled with zeros.

        Notes
        -----
        The algorithm is based on the cholesky decomposition and, therefore,
        in case matrix ``A`` is close to not positive defined, the function
        raises LinalgError.

        Both matrices ``A`` and ``B`` are stored in LAPACK banded storage.

        References
        ----------
        .. [1] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`

        """

        def find_b_inv_elem(i, j, U, D, B):
            rng = min(3, n - i - 1)
            rng_sum = 0.0
            if j == 0:
                for k in range(1, rng + 1):
                    rng_sum -= U[-k - 1, i + k] * B[-k - 1, i + k]
                rng_sum += D[i]
                B[-1, i] = rng_sum
            else:
                for k in range(1, rng + 1):
                    diag = abs(k - j)
                    ind = i + min(k, j)
                    rng_sum -= U[-k - 1, i + k] * B[-diag - 1, ind + diag]
                B[-j - 1, i + j] = rng_sum
        U = cholesky_banded(A)
        for i in range(2, 5):
            U[-i, i - 1:] /= U[-1, :-i + 1]
        D = 1.0 / U[-1] ** 2
        U[-1] /= U[-1]
        n = U.shape[1]
        B = np.zeros(shape=(4, n))
        for i in range(n - 1, -1, -1):
            for j in range(min(3, n - i - 1), -1, -1):
                find_b_inv_elem(i, j, U, D, B)
        B[0] = [0.0] * n
        return B

    def _gcv(lam, X, XtWX, wE, XtE):
        """
        Computes the generalized cross-validation criteria [1].

        Parameters
        ----------
        lam : float, (:math:`\\lambda \\geq 0`)
            Regularization parameter.
        X : array, shape (5, n)
            Matrix is stored in LAPACK banded storage.
        XtWX : array, shape (4, n)
            Product :math:`X^T W X` stored in LAPACK banded storage.
        wE : array, shape (5, n)
            Matrix :math:`W^{-1} E` stored in LAPACK banded storage.
        XtE : array, shape (4, n)
            Product :math:`X^T E` stored in LAPACK banded storage.

        Returns
        -------
        res : float
            Value of the GCV criteria with the regularization parameter
            :math:`\\lambda`.

        Notes
        -----
        Criteria is computed from the formula (1.3.2) [3]:

        .. math:

        GCV(\\lambda) = \\dfrac{1}{n} \\sum\\limits_{k = 1}^{n} \\dfrac{ \\left(
        y_k - f_{\\lambda}(x_k) \\right)^2}{\\left( 1 - \\Tr{A}/n\\right)^2}$.
        The criteria is discussed in section 1.3 [3].

        The numerator is computed using (2.2.4) [3] and the denominator is
        computed using an algorithm from [2] (see in the ``compute_b_inv``
        function).

        References
        ----------
        .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
            for observational data, Philadelphia, Pennsylvania: Society for
            Industrial and Applied Mathematics, 1990, pp. 45-65.
            :doi:`10.1137/1.9781611970128`
        .. [2] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`
        .. [3] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
            BSc thesis, 2022. Might be available (in Russian)
            `here <https://www.hse.ru/ba/am/students/diplomas/620910604>`_

        """
        n = X.shape[1]
        c = solve_banded((2, 2), X + lam * wE, y)
        res = np.zeros(n)
        tmp = wE * c
        for i in range(n):
            for j in range(max(0, i - n + 3), min(5, i + 3)):
                res[i] += tmp[j, i + 2 - j]
        numer = np.linalg.norm(lam * res) ** 2 / n
        lhs = XtWX + lam * XtE
        try:
            b_banded = compute_b_inv(lhs)
            tr = b_banded * XtWX
            tr[:-1] *= 2
            denom = (1 - sum(sum(tr)) / n) ** 2
        except LinAlgError:
            raise ValueError('Seems like the problem is ill-posed')
        res = numer / denom
        return res
    n = X.shape[1]
    XtWX = compute_banded_symmetric_XT_W_Y(X, w, X)
    XtE = compute_banded_symmetric_XT_W_Y(X, w, wE)

    def fun(lam):
        return _gcv(lam, X, XtWX, wE, XtE)
    gcv_est = minimize_scalar(fun, bounds=(0, n), method='Bounded')
    if gcv_est.success:
        return gcv_est.x
    raise ValueError(f'Unable to find minimum of the GCV function: {gcv_est.message}')