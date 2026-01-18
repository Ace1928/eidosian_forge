import sys
import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import interpolate, linalg
from scipy.linalg.lapack import get_lapack_funcs
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..model_selection import check_cv
from ..utils import (  # type: ignore
from ..utils._metadata_requests import (
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, LinearRegression, _preprocess_data
def _lars_path_solver(X, y, Xy=None, Gram=None, n_samples=None, max_iter=500, alpha_min=0, method='lar', copy_X=True, eps=np.finfo(float).eps, copy_Gram=True, verbose=0, return_path=True, return_n_iter=False, positive=False):
    """Compute Least Angle Regression or Lasso path using LARS algorithm [1]

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    X : None or ndarray of shape (n_samples, n_features)
        Input data. Note that if X is None then Gram must be specified,
        i.e., cannot be None or False.

    y : None or ndarray of shape (n_samples,)
        Input targets.

    Xy : array-like of shape (n_features,) or (n_features, n_targets),             default=None
        `Xy = np.dot(X.T, y)` that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    Gram : None, 'auto' or array-like of shape (n_features, n_features),             default=None
        Precomputed Gram matrix `(X' * X)`, if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features.

    n_samples : int or float, default=None
        Equivalent size of sample. If `None`, it will be `n_samples`.

    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, default=True
        If ``False``, ``X`` is overwritten.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_Gram : bool, default=True
        If ``False``, ``Gram`` is overwritten.

    verbose : int, default=0
        Controls output verbosity.

    return_path : bool, default=True
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, default=False
        Whether to return the number of iterations.

    positive : bool, default=False
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (``alphas_[alphas_ > 0.].min()`` when fit_path=True) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.

    Returns
    -------
    alphas : array-like of shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features`` or the
        number of nodes in the path with ``alpha >= alpha_min``, whichever
        is smaller.

    active : array-like of shape (n_alphas,)
        Indices of active variables at the end of the path.

    coefs : array-like of shape (n_features, n_alphas + 1)
        Coefficients along the path

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.

    See Also
    --------
    lasso_path
    LassoLars
    Lars
    LassoLarsCV
    LarsCV
    sklearn.decomposition.sparse_encode

    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_

    """
    if method == 'lar' and positive:
        raise ValueError("Positive constraint not supported for 'lar' coding method.")
    n_samples = n_samples if n_samples is not None else y.size
    if Xy is None:
        Cov = np.dot(X.T, y)
    else:
        Cov = Xy.copy()
    if Gram is None or Gram is False:
        Gram = None
        if X is None:
            raise ValueError('X and Gram cannot both be unspecified.')
    elif isinstance(Gram, str) and Gram == 'auto' or Gram is True:
        if Gram is True or X.shape[0] > X.shape[1]:
            Gram = np.dot(X.T, X)
        else:
            Gram = None
    elif copy_Gram:
        Gram = Gram.copy()
    if Gram is None:
        n_features = X.shape[1]
    else:
        n_features = Cov.shape[0]
        if Gram.shape != (n_features, n_features):
            raise ValueError('The shapes of the inputs Gram and Xy do not match.')
    if copy_X and X is not None and (Gram is None):
        X = X.copy('F')
    max_features = min(max_iter, n_features)
    dtypes = set((a.dtype for a in (X, y, Xy, Gram) if a is not None))
    if len(dtypes) == 1:
        return_dtype = next(iter(dtypes))
    else:
        return_dtype = np.float64
    if return_path:
        coefs = np.zeros((max_features + 1, n_features), dtype=return_dtype)
        alphas = np.zeros(max_features + 1, dtype=return_dtype)
    else:
        coef, prev_coef = (np.zeros(n_features, dtype=return_dtype), np.zeros(n_features, dtype=return_dtype))
        alpha, prev_alpha = (np.array([0.0], dtype=return_dtype), np.array([0.0], dtype=return_dtype))
    n_iter, n_active = (0, 0)
    active, indices = (list(), np.arange(n_features))
    sign_active = np.empty(max_features, dtype=np.int8)
    drop = False
    if Gram is None:
        L = np.empty((max_features, max_features), dtype=X.dtype)
        swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (X,))
    else:
        L = np.empty((max_features, max_features), dtype=Gram.dtype)
        swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (Cov,))
    solve_cholesky, = get_lapack_funcs(('potrs',), (L,))
    if verbose:
        if verbose > 1:
            print('Step\t\tAdded\t\tDropped\t\tActive set size\t\tC')
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    tiny32 = np.finfo(np.float32).tiny
    cov_precision = np.finfo(Cov.dtype).precision
    equality_tolerance = np.finfo(np.float32).eps
    if Gram is not None:
        Gram_copy = Gram.copy()
        Cov_copy = Cov.copy()
    while True:
        if Cov.size:
            if positive:
                C_idx = np.argmax(Cov)
            else:
                C_idx = np.argmax(np.abs(Cov))
            C_ = Cov[C_idx]
            if positive:
                C = C_
            else:
                C = np.fabs(C_)
        else:
            C = 0.0
        if return_path:
            alpha = alphas[n_iter, np.newaxis]
            coef = coefs[n_iter]
            prev_alpha = alphas[n_iter - 1, np.newaxis]
            prev_coef = coefs[n_iter - 1]
        alpha[0] = C / n_samples
        if alpha[0] <= alpha_min + equality_tolerance:
            if abs(alpha[0] - alpha_min) > equality_tolerance:
                if n_iter > 0:
                    ss = (prev_alpha[0] - alpha_min) / (prev_alpha[0] - alpha[0])
                    coef[:] = prev_coef + ss * (coef - prev_coef)
                alpha[0] = alpha_min
            if return_path:
                coefs[n_iter] = coef
            break
        if n_iter >= max_iter or n_active >= n_features:
            break
        if not drop:
            if positive:
                sign_active[n_active] = np.ones_like(C_)
            else:
                sign_active[n_active] = np.sign(C_)
            m, n = (n_active, C_idx + n_active)
            Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
            indices[n], indices[m] = (indices[m], indices[n])
            Cov_not_shortened = Cov
            Cov = Cov[1:]
            if Gram is None:
                X.T[n], X.T[m] = swap(X.T[n], X.T[m])
                c = nrm2(X.T[n_active]) ** 2
                L[n_active, :n_active] = np.dot(X.T[n_active], X.T[:n_active].T)
            else:
                Gram[m], Gram[n] = swap(Gram[m], Gram[n])
                Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
                c = Gram[n_active, n_active]
                L[n_active, :n_active] = Gram[n_active, :n_active]
            if n_active:
                linalg.solve_triangular(L[:n_active, :n_active], L[n_active, :n_active], trans=0, lower=1, overwrite_b=True, **SOLVE_TRIANGULAR_ARGS)
            v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
            diag = max(np.sqrt(np.abs(c - v)), eps)
            L[n_active, n_active] = diag
            if diag < 1e-07:
                warnings.warn('Regressors in active set degenerate. Dropping a regressor, after %i iterations, i.e. alpha=%.3e, with an active set of %i regressors, and the smallest cholesky pivot element being %.3e. Reduce max_iter or increase eps parameters.' % (n_iter, alpha.item(), n_active, diag), ConvergenceWarning)
                Cov = Cov_not_shortened
                Cov[0] = 0
                Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
                continue
            active.append(indices[n_active])
            n_active += 1
            if verbose > 1:
                print('%s\t\t%s\t\t%s\t\t%s\t\t%s' % (n_iter, active[-1], '', n_active, C))
        if method == 'lasso' and n_iter > 0 and (prev_alpha[0] < alpha[0]):
            warnings.warn('Early stopping the lars path, as the residues are small and the current value of alpha is no longer well controlled. %i iterations, alpha=%.3e, previous alpha=%.3e, with an active set of %i regressors.' % (n_iter, alpha.item(), prev_alpha.item(), n_active), ConvergenceWarning)
            break
        least_squares, _ = solve_cholesky(L[:n_active, :n_active], sign_active[:n_active], lower=True)
        if least_squares.size == 1 and least_squares == 0:
            least_squares[...] = 1
            AA = 1.0
        else:
            AA = 1.0 / np.sqrt(np.sum(least_squares * sign_active[:n_active]))
            if not np.isfinite(AA):
                i = 0
                L_ = L[:n_active, :n_active].copy()
                while not np.isfinite(AA):
                    L_.flat[::n_active + 1] += 2 ** i * eps
                    least_squares, _ = solve_cholesky(L_, sign_active[:n_active], lower=True)
                    tmp = max(np.sum(least_squares * sign_active[:n_active]), eps)
                    AA = 1.0 / np.sqrt(tmp)
                    i += 1
            least_squares *= AA
        if Gram is None:
            eq_dir = np.dot(X.T[:n_active].T, least_squares)
            corr_eq_dir = np.dot(X.T[n_active:], eq_dir)
        else:
            corr_eq_dir = np.dot(Gram[:n_active, n_active:].T, least_squares)
        np.around(corr_eq_dir, decimals=cov_precision, out=corr_eq_dir)
        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        if positive:
            gamma_ = min(g1, C / AA)
        else:
            g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
            gamma_ = min(g1, g2, C / AA)
        drop = False
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            idx = np.where(z == z_pos)[0][::-1]
            sign_active[idx] = -sign_active[idx]
            if method == 'lasso':
                gamma_ = z_pos
            drop = True
        n_iter += 1
        if return_path:
            if n_iter >= coefs.shape[0]:
                del coef, alpha, prev_alpha, prev_coef
                add_features = 2 * max(1, max_features - n_active)
                coefs = np.resize(coefs, (n_iter + add_features, n_features))
                coefs[-add_features:] = 0
                alphas = np.resize(alphas, n_iter + add_features)
                alphas[-add_features:] = 0
            coef = coefs[n_iter]
            prev_coef = coefs[n_iter - 1]
        else:
            prev_coef = coef
            prev_alpha[0] = alpha[0]
            coef = np.zeros_like(coef)
        coef[active] = prev_coef[active] + gamma_ * least_squares
        Cov -= gamma_ * corr_eq_dir
        if drop and method == 'lasso':
            for ii in idx:
                arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii)
            n_active -= 1
            drop_idx = [active.pop(ii) for ii in idx]
            if Gram is None:
                for ii in idx:
                    for i in range(ii, n_active):
                        X.T[i], X.T[i + 1] = swap(X.T[i], X.T[i + 1])
                        indices[i], indices[i + 1] = (indices[i + 1], indices[i])
                residual = y - np.dot(X[:, :n_active], coef[active])
                temp = np.dot(X.T[n_active], residual)
                Cov = np.r_[temp, Cov]
            else:
                for ii in idx:
                    for i in range(ii, n_active):
                        indices[i], indices[i + 1] = (indices[i + 1], indices[i])
                        Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i + 1])
                        Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i], Gram[:, i + 1])
                temp = Cov_copy[drop_idx] - np.dot(Gram_copy[drop_idx], coef)
                Cov = np.r_[temp, Cov]
            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.0)
            if verbose > 1:
                print('%s\t\t%s\t\t%s\t\t%s\t\t%s' % (n_iter, '', drop_idx, n_active, abs(temp)))
    if return_path:
        alphas = alphas[:n_iter + 1]
        coefs = coefs[:n_iter + 1]
        if return_n_iter:
            return (alphas, active, coefs.T, n_iter)
        else:
            return (alphas, active, coefs.T)
    elif return_n_iter:
        return (alpha, active, coef, n_iter)
    else:
        return (alpha, active, coef)