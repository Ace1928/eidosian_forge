from numbers import Integral
import numpy as np
from joblib import effective_n_jobs
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..metrics import check_scoring
from ..model_selection import check_cv
from ..model_selection._validation import _score
from ..utils._param_validation import HasMethods, Interval, RealNotInt
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split, available_if
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin, _get_feature_importances
class RFECV(RFE):
    """Recursive feature elimination with cross-validation to select features.

    The number of features selected is tuned automatically by fitting an :class:`RFE`
    selector on the different cross-validation splits (provided by the `cv` parameter).
    The performance of the :class:`RFE` selector are evaluated using `scorer` for
    different number of selected features and aggregated together. Finally, the scores
    are averaged across folds and the number of features selected is set to the number
    of features that maximize the cross-validation score.
    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.

    min_features_to_select : int, default=1
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and ``min_features_to_select`` isn't divisible by
        ``step``.

        .. versionadded:: 0.20

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is a classifier or if ``y`` is neither binary nor multiclass,
        :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value of None changed from 3-fold to 5-fold.

    scoring : str, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int or None, default=None
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance.
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a classifier.

    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.

    cv_results_ : dict of ndarrays
        A dict with keys:

        split(k)_test_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.

        mean_test_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.

        std_test_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.

        .. versionadded:: 1.0

    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    ranking_ : narray of shape (n_features,)
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    See Also
    --------
    RFE : Recursive feature elimination.

    Notes
    -----
    The size of all values in ``cv_results_`` is equal to
    ``ceil((n_features - min_features_to_select) / step) + 1``,
    where step is the number of features removed at each iteration.

    Allows NaN/Inf in the input if the underlying estimator does as well.

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFECV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
    """
    _parameter_constraints: dict = {**RFE._parameter_constraints, 'min_features_to_select': [Interval(Integral, 0, None, closed='neither')], 'cv': ['cv_object'], 'scoring': [None, str, callable], 'n_jobs': [None, Integral]}
    _parameter_constraints.pop('n_features_to_select')

    def __init__(self, estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None, verbose=0, n_jobs=None, importance_getter='auto'):
        self.estimator = estimator
        self.step = step
        self.importance_getter = importance_getter
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_features_to_select = min_features_to_select

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, groups=None):
        """Fit the RFE model and automatically tune the number of selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the total number of features.

        y : array-like of shape (n_samples,)
            Target values (integers for classification, real numbers for
            regression).

        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _raise_for_unsupported_routing(self, 'fit', groups=groups)
        X, y = self._validate_data(X, y, accept_sparse='csr', ensure_min_features=2, force_all_finite=False, multi_output=True)
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]
        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        rfe = RFE(estimator=self.estimator, n_features_to_select=self.min_features_to_select, importance_getter=self.importance_getter, step=self.step, verbose=self.verbose)
        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = (list, _rfe_single_fit)
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)
        scores = parallel((func(rfe, self.estimator, X, y, train, test, scorer) for train, test in cv.split(X, y, groups)))
        scores = np.array(scores)
        scores_sum = np.sum(scores, axis=0)
        scores_sum_rev = scores_sum[::-1]
        argmax_idx = len(scores_sum) - np.argmax(scores_sum_rev) - 1
        n_features_to_select = max(n_features - argmax_idx * step, self.min_features_to_select)
        rfe = RFE(estimator=self.estimator, n_features_to_select=n_features_to_select, step=self.step, importance_getter=self.importance_getter, verbose=self.verbose)
        rfe.fit(X, y)
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._transform(X), y)
        scores_rev = scores[:, ::-1]
        self.cv_results_ = {}
        self.cv_results_['mean_test_score'] = np.mean(scores_rev, axis=0)
        self.cv_results_['std_test_score'] = np.std(scores_rev, axis=0)
        for i in range(scores.shape[0]):
            self.cv_results_[f'split{i}_test_score'] = scores_rev[i]
        return self