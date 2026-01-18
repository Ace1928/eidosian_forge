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
class MultiTaskLasso(MultiTaskElasticNet):
    """Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <multi_task_lasso>`.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1/L2 term. Defaults to 1.0.

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
    coef_ : ndarray of shape (n_targets, n_features)
        Parameter vector (W in the cost function formula).
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    intercept_ : ndarray of shape (n_targets,)
        Independent term in decision function.

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    dual_gap_ : ndarray of shape (n_alphas,)
        The dual gaps at the end of the optimization for each alpha.

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
    Lasso: Linear Model trained with L1 prior as regularizer (aka the Lasso).
    MultiTaskLassoCV: Multi-task L1 regularized linear model with built-in
        cross-validation.
    MultiTaskElasticNetCV: Multi-task L1/L2 ElasticNet with built-in cross-validation.

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X and y arguments of the fit
    method should be directly passed as Fortran-contiguous numpy arrays.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.MultiTaskLasso(alpha=0.1)
    >>> clf.fit([[0, 1], [1, 2], [2, 4]], [[0, 0], [1, 1], [2, 3]])
    MultiTaskLasso(alpha=0.1)
    >>> print(clf.coef_)
    [[0.         0.60809415]
    [0.         0.94592424]]
    >>> print(clf.intercept_)
    [-0.41888636 -0.87382323]
    """
    _parameter_constraints: dict = {**MultiTaskElasticNet._parameter_constraints}
    _parameter_constraints.pop('l1_ratio')

    def __init__(self, alpha=1.0, *, fit_intercept=True, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, random_state=None, selection='cyclic'):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.tol = tol
        self.warm_start = warm_start
        self.l1_ratio = 1.0
        self.random_state = random_state
        self.selection = selection