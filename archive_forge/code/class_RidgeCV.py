import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse import linalg as sp_linalg
from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier
from ..exceptions import ConvergenceWarning
from ..metrics import check_scoring, get_scorer_names
from ..model_selection import GridSearchCV
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import _sparse_linalg_cg
from ..utils.metadata_routing import (
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data
from ._sag import sag_solver
class RidgeCV(_RoutingNotSupportedMixin, MultiOutputMixin, RegressorMixin, _BaseRidgeCV):
    """Ridge regression with built-in cross-validation.

    See glossary entry for :term:`cross-validation estimator`.

    By default, it performs efficient Leave-One-Out Cross-Validation.

    Read more in the :ref:`User Guide <ridge_regression>`.

    Parameters
    ----------
    alphas : array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)
        Array of alpha values to try.
        Regularization strength; must be a positive float. Regularization
        improves the conditioning of the problem and reduces the variance of
        the estimates. Larger values specify stronger regularization.
        Alpha corresponds to ``1 / (2C)`` in other linear models such as
        :class:`~sklearn.linear_model.LogisticRegression` or
        :class:`~sklearn.svm.LinearSVC`.
        If using Leave-One-Out cross-validation, alphas must be positive.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    scoring : str, callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the negative mean squared error if cv is 'auto' or None
        (i.e. when using leave-one-out cross-validation), and r2 score
        otherwise.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the efficient Leave-One-Out cross-validation
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    gcv_mode : {'auto', 'svd', 'eigen'}, default='auto'
        Flag indicating which strategy to use when performing
        Leave-One-Out Cross-Validation. Options are::

            'auto' : use 'svd' if n_samples > n_features, otherwise use 'eigen'
            'svd' : force use of singular value decomposition of X when X is
                dense, eigenvalue decomposition of X^T.X when X is sparse.
            'eigen' : force computation via eigendecomposition of X.X^T

        The 'auto' mode is the default and is intended to pick the cheaper
        option of the two depending on the shape of the training data.

    store_cv_values : bool, default=False
        Flag indicating if the cross-validation values corresponding to
        each alpha should be stored in the ``cv_values_`` attribute (see
        below). This flag is only compatible with ``cv=None`` (i.e. using
        Leave-One-Out Cross-Validation).

    alpha_per_target : bool, default=False
        Flag indicating whether to optimize the alpha value (picked from the
        `alphas` parameter list) for each target separately (for multi-output
        settings: multiple prediction targets). When set to `True`, after
        fitting, the `alpha_` attribute will contain a value for each target.
        When set to `False`, a single alpha is used for all targets.

        .. versionadded:: 0.24

    Attributes
    ----------
    cv_values_ : ndarray of shape (n_samples, n_alphas) or             shape (n_samples, n_targets, n_alphas), optional
        Cross-validation values for each alpha (only available if
        ``store_cv_values=True`` and ``cv=None``). After ``fit()`` has been
        called, this attribute will contain the mean squared errors if
        `scoring is None` otherwise it will contain standardized per point
        prediction values.

    coef_ : ndarray of shape (n_features) or (n_targets, n_features)
        Weight vector(s).

    intercept_ : float or ndarray of shape (n_targets,)
        Independent term in decision function. Set to 0.0 if
        ``fit_intercept = False``.

    alpha_ : float or ndarray of shape (n_targets,)
        Estimated regularization parameter, or, if ``alpha_per_target=True``,
        the estimated regularization parameter for each target.

    best_score_ : float or ndarray of shape (n_targets,)
        Score of base estimator with best alpha, or, if
        ``alpha_per_target=True``, a score for each target.

        .. versionadded:: 0.23

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    Ridge : Ridge regression.
    RidgeClassifier : Classifier based on ridge regression on {-1, 1} labels.
    RidgeClassifierCV : Ridge classifier with built-in cross validation.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.linear_model import RidgeCV
    >>> X, y = load_diabetes(return_X_y=True)
    >>> clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X, y)
    >>> clf.score(X, y)
    0.5166...
    """

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model with cv.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data. If using GCV, will be cast to float64
            if necessary.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        When sample_weight is provided, the selected hyperparameter may depend
        on whether we use leave-one-out cross-validation (cv=None or cv='auto')
        or another form of cross-validation, because only leave-one-out
        cross-validation takes the sample weights into account when computing
        the validation score.
        """
        _raise_for_unsupported_routing(self, 'fit', sample_weight=sample_weight)
        super().fit(X, y, sample_weight=sample_weight)
        return self