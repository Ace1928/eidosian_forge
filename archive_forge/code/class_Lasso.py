import numbers
import sys
import warnings
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import sparse
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..model_selection import check_cv
from ..utils import Bunch, check_array, check_scalar
from ..utils._metadata_requests import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import safe_sparse_dot
from ..utils.metadata_routing import (
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import _cd_fast as cd_fast  # type: ignore
from ._base import LinearModel, _pre_fit, _preprocess_data
class Lasso(ElasticNet):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso).

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

    Read more in the :ref:`User Guide <lasso>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

        When `alpha = 0`, the objective is equivalent to ordinary least
        squares, solved by the :class:`LinearRegression` object. For numerical
        reasons, using `alpha = 0` with the `Lasso` object is not advised.
        Instead, you should use the :class:`LinearRegression` object.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : bool or array-like of shape (n_features, n_features),                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``, see Notes below.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the cost function formula).

    dual_gap_ : float or ndarray of shape (n_targets,)
        Given param alpha, the dual gaps at the end of the optimization,
        same shape as each observation of y.

    sparse_coef_ : sparse matrix of shape (n_features, 1) or             (n_targets, n_features)
        Readonly property derived from ``coef_``.

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : int or list of int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    lars_path : Regularization path using LARS.
    lasso_path : Regularization path using Lasso.
    LassoLars : Lasso Path along the regularization parameter using LARS algorithm.
    LassoCV : Lasso alpha parameter by cross-validation.
    LassoLarsCV : Lasso least angle parameter algorithm by cross-validation.
    sklearn.decomposition.sparse_encode : Sparse coding array estimator.

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.

    Regularization improves the conditioning of the problem and
    reduces the variance of the estimates. Larger values specify stronger
    regularization. Alpha corresponds to `1 / (2C)` in other linear
    models such as :class:`~sklearn.linear_model.LogisticRegression` or
    :class:`~sklearn.svm.LinearSVC`. If an array is passed, penalties are
    assumed to be specific to the targets. Hence they must correspond in
    number.

    The precise stopping criteria based on `tol` are the following: First, check that
    that maximum coordinate update, i.e. :math:`\\max_j |w_j^{new} - w_j^{old}|`
    is smaller than `tol` times the maximum absolute coefficient, :math:`\\max_j |w_j|`.
    If so, then additionally check whether the dual gap is smaller than `tol` times
    :math:`||y||_2^2 / n_{\\text{samples}}`.

    The target can be a 2-dimensional array, resulting in the optimization of the
    following objective::

        (1 / (2 * n_samples)) * ||Y - XW||^2_F + alpha * ||W||_11

    where :math:`||W||_{1,1}` is the sum of the magnitude of the matrix coefficients.
    It should not be confused with :class:`~sklearn.linear_model.MultiTaskLasso` which
    instead penalizes the :math:`L_{2,1}` norm of the coefficients, yielding row-wise
    sparsity in the coefficients.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.Lasso(alpha=0.1)
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    Lasso(alpha=0.1)
    >>> print(clf.coef_)
    [0.85 0.  ]
    >>> print(clf.intercept_)
    0.15...
    """
    _parameter_constraints: dict = {**ElasticNet._parameter_constraints}
    _parameter_constraints.pop('l1_ratio')
    path = staticmethod(enet_path)

    def __init__(self, alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
        super().__init__(alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept, precompute=precompute, copy_X=copy_X, max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection)