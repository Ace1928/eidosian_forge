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
class MultiTaskElasticNetCV(RegressorMixin, LinearModelCV):
    """Multi-task L1/L2 ElasticNet with built-in cross-validation.

    See glossary entry for :term:`cross-validation estimator`.

    The optimization objective for MultiTaskElasticNet is::

        (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
        + alpha * l1_ratio * ||W||_21
        + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

    Where::

        ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}

    i.e. the sum of norm of each row.

    Read more in the :ref:`User Guide <multi_task_elastic_net>`.

    .. versionadded:: 0.15

    Parameters
    ----------
    l1_ratio : float or list of float, default=0.5
        The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.
        For l1_ratio = 1 the penalty is an L1/L2 penalty. For l1_ratio = 0 it
        is an L2 penalty.
        For ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.
        This parameter can be a list, in which case the different
        values are tested by cross-validation and the one giving the best
        prediction score is used. Note that a good choice of list of
        values for l1_ratio is often to put more values close to 1
        (i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,
        .9, .95, .99, 1]``.

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, default=100
        Number of alphas along the regularization path.

    alphas : array-like, default=None
        List of alphas where to compute the models.
        If not provided, set automatically.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    max_iter : int, default=1000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    verbose : bool or int, default=0
        Amount of verbosity.

    n_jobs : int, default=None
        Number of CPUs to use during the cross validation. Note that this is
        used only if multiple values for l1_ratio are given.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

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
        Parameter vector (W in the cost function formula).
        Note that ``coef_`` stores the transpose of ``W``, ``W.T``.

    alpha_ : float
        The amount of penalization chosen by cross validation.

    mse_path_ : ndarray of shape (n_alphas, n_folds) or                 (n_l1_ratio, n_alphas, n_folds)
        Mean square error for the test set on each fold, varying alpha.

    alphas_ : ndarray of shape (n_alphas,) or (n_l1_ratio, n_alphas)
        The grid of alphas used for fitting, for each l1_ratio.

    l1_ratio_ : float
        Best l1_ratio obtained by cross-validation.

    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance for the optimal alpha.

    dual_gap_ : float
        The dual gap at the end of the optimization for the optimal alpha.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MultiTaskElasticNet : Multi-task L1/L2 ElasticNet with built-in cross-validation.
    ElasticNetCV : Elastic net model with best model selection by
        cross-validation.
    MultiTaskLassoCV : Multi-task Lasso model trained with L1 norm
        as regularizer and built-in cross-validation.

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    In `fit`, once the best parameters `l1_ratio` and `alpha` are found through
    cross-validation, the model is fit again using the entire training set.

    To avoid unnecessary memory duplication the `X` and `y` arguments of the
    `fit` method should be directly passed as Fortran-contiguous numpy arrays.

    Examples
    --------
    >>> from sklearn import linear_model
    >>> clf = linear_model.MultiTaskElasticNetCV(cv=3)
    >>> clf.fit([[0,0], [1, 1], [2, 2]],
    ...         [[0, 0], [1, 1], [2, 2]])
    MultiTaskElasticNetCV(cv=3)
    >>> print(clf.coef_)
    [[0.52875032 0.46958558]
     [0.52875032 0.46958558]]
    >>> print(clf.intercept_)
    [0.00166409 0.00166409]
    """
    _parameter_constraints: dict = {**LinearModelCV._parameter_constraints, 'l1_ratio': [Interval(Real, 0, 1, closed='both'), 'array-like']}
    _parameter_constraints.pop('precompute')
    _parameter_constraints.pop('positive')
    path = staticmethod(enet_path)

    def __init__(self, *, l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, max_iter=1000, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=None, random_state=None, selection='cyclic'):
        self.l1_ratio = l1_ratio
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.selection = selection

    def _get_estimator(self):
        return MultiTaskElasticNet()

    def _is_multitask(self):
        return True

    def _more_tags(self):
        return {'multioutput_only': True}

    def fit(self, X, y, **params):
        """Fit MultiTaskElasticNet model with coordinate descent.

        Fit is on grid of alphas and best alpha estimated by cross-validation.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples, n_targets)
            Training target variable. Will be cast to X's dtype if necessary.

        **params : dict, default=None
            Parameters to be passed to the CV splitter.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`,
                which can be set by using
                ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns MultiTaskElasticNet instance.
        """
        return super().fit(X, y, **params)