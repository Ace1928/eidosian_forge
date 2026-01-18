import array
import itertools
import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from .base import (
from .metrics.pairwise import pairwise_distances_argmin
from .preprocessing import LabelBinarizer
from .utils import check_random_state
from .utils._param_validation import HasMethods, Interval
from .utils._tags import _safe_tags
from .utils.metadata_routing import (
from .utils.metaestimators import _safe_split, available_if
from .utils.multiclass import (
from .utils.parallel import Parallel, delayed
from .utils.validation import _check_method_params, _num_samples, check_is_fitted
class OutputCodeClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """(Error-Correcting) Output-Code multiclass strategy.

    Output-code based strategies consist in representing each class with a
    binary code (an array of 0s and 1s). At fitting time, one binary
    classifier per bit in the code book is fitted.  At prediction time, the
    classifiers are used to project new points in the class space and the class
    closest to the points is chosen. The main advantage of these strategies is
    that the number of classifiers used can be controlled by the user, either
    for compressing the model (0 < `code_size` < 1) or for making the model more
    robust to errors (`code_size` > 1). See the documentation for more details.

    Read more in the :ref:`User Guide <ecoc>`.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and one of
        :term:`decision_function` or :term:`predict_proba`.

    code_size : float, default=1.5
        Percentage of the number of classes to be used to create the code book.
        A number between 0 and 1 will require fewer classifiers than
        one-vs-the-rest. A number greater than 1 will require more classifiers
        than one-vs-the-rest.

    random_state : int, RandomState instance, default=None
        The generator used to initialize the codebook.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the multiclass problems
        are computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    estimators_ : list of `int(n_classes * code_size)` estimators
        Estimators used for predictions.

    classes_ : ndarray of shape (n_classes,)
        Array containing labels.

    code_book_ : ndarray of shape (n_classes, `len(estimators_)`)
        Binary array containing the code of each class.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    OneVsRestClassifier : One-vs-all multiclass strategy.
    OneVsOneClassifier : One-vs-one multiclass strategy.

    References
    ----------

    .. [1] "Solving multiclass learning problems via error-correcting output
       codes",
       Dietterich T., Bakiri G.,
       Journal of Artificial Intelligence Research 2,
       1995.

    .. [2] "The error coding method and PICTs",
       James G., Hastie T.,
       Journal of Computational and Graphical statistics 7,
       1998.

    .. [3] "The Elements of Statistical Learning",
       Hastie T., Tibshirani R., Friedman J., page 606 (second-edition)
       2008.

    Examples
    --------
    >>> from sklearn.multiclass import OutputCodeClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = OutputCodeClassifier(
    ...     estimator=RandomForestClassifier(random_state=0),
    ...     random_state=0).fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    """
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit', 'decision_function']), HasMethods(['fit', 'predict_proba'])], 'code_size': [Interval(Real, 0.0, None, closed='neither')], 'random_state': ['random_state'], 'n_jobs': [Integral, None]}

    def __init__(self, estimator, *, code_size=1.5, random_state=None, n_jobs=None):
        self.estimator = estimator
        self.code_size = code_size
        self.random_state = random_state
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, **fit_params):
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Multi-class targets.

        **fit_params : dict
            Parameters passed to the ``estimator.fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        _raise_for_params(fit_params, self, 'fit')
        routed_params = process_routing(self, 'fit', **fit_params)
        y = self._validate_data(X='no_validation', y=y)
        random_state = check_random_state(self.random_state)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        if n_classes == 0:
            raise ValueError('OutputCodeClassifier can not be fit when no class is present.')
        n_estimators = int(n_classes * self.code_size)
        self.code_book_ = random_state.uniform(size=(n_classes, n_estimators))
        self.code_book_[self.code_book_ > 0.5] = 1.0
        if hasattr(self.estimator, 'decision_function'):
            self.code_book_[self.code_book_ != 1] = -1.0
        else:
            self.code_book_[self.code_book_ != 1] = 0.0
        classes_index = {c: i for i, c in enumerate(self.classes_)}
        Y = np.array([self.code_book_[classes_index[y[i]]] for i in range(_num_samples(y))], dtype=int)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_fit_binary)(self.estimator, X, Y[:, i], fit_params=routed_params.estimator.fit) for i in range(Y.shape[1])))
        if hasattr(self.estimators_[0], 'n_features_in_'):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], 'feature_names_in_'):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_
        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted multi-class targets.
        """
        check_is_fitted(self)
        Y = np.array([_predict_binary(e, X) for e in self.estimators_], order='F', dtype=np.float64).T
        pred = pairwise_distances_argmin(Y, self.code_book_, metric='euclidean')
        return self.classes_[pred]

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add(estimator=self.estimator, method_mapping=MethodMapping().add(callee='fit', caller='fit'))
        return router