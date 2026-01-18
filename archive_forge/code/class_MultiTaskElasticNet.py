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
class MultiTaskElasticNet(Lasso):
    """Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer.

    The optimization objective for MultiTaskElasticNet is::

        (1 / (2 * n_samples)) * ||Y - XW||_Fro^2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = sum_i sqrt(sum_j W_ij ^ 2)

    i.e. the sum of norms of each row.

    Read more in the :ref:`User Guide <multi_task_elastic_net>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1/L2 term. Defaults to 1.0.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
        For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
        is an L2 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

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
    intercept_ : ndarray of shape (n_targets,)
        Independent term in decision function.

    coef_ : ndarray of shape (n_targets, n_features)
        Parameter vector (W in the cost function formula). If a 1D y is
        passed in at fit (non multi-task usage), ``coef_`` is then a 1D array.
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    dual_gap_ : float
        The dual gaps at the end of the optimization.

    eps_ : float
        The tolerance scaled scaled by the variance of the target `y`.

    sparse_coef_ : sparse matrix of shape (n_features,) or             (n_targets, n_features)
        Sparse representation of the `coef_`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MultiTaskElasticNetCV : Multi-task L1/L2 ElasticNet with built-in
        cross-validation.
    ElasticNet : Linear regression with combined L1 and L2 priors as regularizer.
    MultiTaskLasso : Multi-task Lasso model trained with L1/L2
        mixed-norm as regularizer.

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X and y arguments of the fit
    method should be directly passed as Fortran-contiguous numpy arrays.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.MultiTaskElasticNet(alpha=0.1)
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
    MultiTaskElasticNet(alpha=0.1)
    >>> print(clf.coef_)
    [[0.45663524 0.45612256]
     [0.45663524 0.45612256]]
    >>> print(clf.intercept_)
    [0.0872422 0.0872422]
    """
    _parameter_constraints: dict = {**ElasticNet._parameter_constraints}
    for param in ('precompute', 'positive'):
        _parameter_constraints.pop(param)

    def __init__(self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, random_state=None, selection='cyclic'):
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit MultiTaskElasticNet model with coordinate descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data.
        y : ndarray of shape (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.

        To avoid memory re-allocation it is advised to allocate the
        initial data in memory directly using that format.
        """
        check_X_params = dict(dtype=[np.float64, np.float32], order='F', copy=self.copy_X and self.fit_intercept)
        check_y_params = dict(ensure_2d=False, order='F')
        X, y = self._validate_data(X, y, validate_separately=(check_X_params, check_y_params))
        check_consistent_length(X, y)
        y = y.astype(X.dtype)
        if hasattr(self, 'l1_ratio'):
            model_str = 'ElasticNet'
        else:
            model_str = 'Lasso'
        if y.ndim == 1:
            raise ValueError('For mono-task outputs, use %s' % model_str)
        n_samples, n_features = X.shape
        n_targets = y.shape[1]
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y, fit_intercept=self.fit_intercept, copy=False)
        if not self.warm_start or not hasattr(self, 'coef_'):
            self.coef_ = np.zeros((n_targets, n_features), dtype=X.dtype.type, order='F')
        l1_reg = self.alpha * self.l1_ratio * n_samples
        l2_reg = self.alpha * (1.0 - self.l1_ratio) * n_samples
        self.coef_ = np.asfortranarray(self.coef_)
        random = self.selection == 'random'
        self.coef_, self.dual_gap_, self.eps_, self.n_iter_ = cd_fast.enet_coordinate_descent_multi_task(self.coef_, l1_reg, l2_reg, X, y, self.max_iter, self.tol, check_random_state(self.random_state), random)
        self.dual_gap_ /= n_samples
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def _more_tags(self):
        return {'multioutput_only': True}