from abc import abstractmethod
from numbers import Integral
import numpy as np
from ..base import (
from ..exceptions import NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._param_validation import StrOptions
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
class VotingRegressor(_RoutingNotSupportedMixin, RegressorMixin, _BaseVoting):
    """Prediction voting regressor for unfitted estimators.

    A voting regressor is an ensemble meta-estimator that fits several base
    regressors, each on the whole dataset. Then it averages the individual
    predictions to form a final prediction.

    Read more in the :ref:`User Guide <voting_regressor>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        :meth:`set_params`.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    weights : array-like of shape (n_regressors,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying regressor exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    VotingClassifier : Soft Voting/Majority Rule classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> r3 = KNeighborsRegressor()
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])
    >>> print(er.fit(X, y).predict(X))
    [ 6.8...  8.4... 12.5... 17.8... 26...  34...]

    In the following example, we drop the `'lr'` estimator with
    :meth:`~VotingRegressor.set_params` and fit the remaining two estimators:

    >>> er = er.set_params(lr='drop')
    >>> er = er.fit(X, y)
    >>> len(er.estimators_)
    2
    """

    def __init__(self, estimators, *, weights=None, n_jobs=None, verbose=False):
        super().__init__(estimators=estimators)
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _raise_for_unsupported_routing(self, 'fit', sample_weight=sample_weight)
        y = column_or_1d(y, warn=True)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        return np.average(self._predict(X), axis=1, weights=self._weights_not_none)

    def transform(self, X):
        """Return predictions for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_classifiers)
            Values predicted by each regressor.
        """
        check_is_fitted(self)
        return self._predict(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present here for API consistency by convention.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, 'n_features_in_')
        _check_feature_names_in(self, input_features, generate_names=False)
        class_name = self.__class__.__name__.lower()
        return np.asarray([f'{class_name}_{name}' for name, est in self.estimators if est != 'drop'], dtype=object)