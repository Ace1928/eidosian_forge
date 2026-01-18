import warnings
from numbers import Integral
import numpy as np
from sklearn.neighbors._base import _check_precomputed
from ..base import ClassifierMixin, _fit_context
from ..metrics._pairwise_distances_reduction import (
from ..utils._param_validation import StrOptions
from ..utils.arrayfuncs import _all_with_any_reduction_axis_1
from ..utils.extmath import weighted_mode
from ..utils.fixes import _mode
from ..utils.validation import _is_arraylike, _num_samples, check_is_fitted
from ._base import KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin, _get_weights
class KNeighborsClassifier(KNeighborsMixin, ClassifierMixin, NeighborsBase):
    """Classifier implementing the k-nearest neighbors vote.

    Read more in the :ref:`User Guide <classification>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    weights : {'uniform', 'distance'}, callable or None, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

        Refer to the example entitled
        :ref:`sphx_glr_auto_examples_neighbors_plot_classification.py`
        showing the impact of the `weights` parameter on the decision
        boundary.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    p : float, default=2
        Power parameter for the Minkowski metric. When p = 1, this is equivalent
        to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
        For arbitrary p, minkowski_distance (l_p) is used. This parameter is expected
        to be positive.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit. X may be a :term:`sparse graph`, in which
        case only "nonzero" elements may be considered neighbors.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Doesn't affect :meth:`fit` method.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Class labels known to the classifier

    effective_metric_ : str or callble
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    outputs_2d_ : bool
        False when `y`'s shape is (n_samples, ) or (n_samples, 1) during fit
        otherwise True.

    See Also
    --------
    RadiusNeighborsClassifier: Classifier based on neighbors within a fixed radius.
    KNeighborsRegressor: Regression based on k-nearest neighbors.
    RadiusNeighborsRegressor: Regression based on neighbors within a fixed radius.
    NearestNeighbors: Unsupervised learner for implementing neighbor searches.

    Notes
    -----
    See :ref:`Nearest Neighbors <neighbors>` in the online documentation
    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

    .. warning::

       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances
       but different labels, the results will depend on the ordering of the
       training data.

    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

    Examples
    --------
    >>> X = [[0], [1], [2], [3]]
    >>> y = [0, 0, 1, 1]
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> neigh = KNeighborsClassifier(n_neighbors=3)
    >>> neigh.fit(X, y)
    KNeighborsClassifier(...)
    >>> print(neigh.predict([[1.1]]))
    [0]
    >>> print(neigh.predict_proba([[0.9]]))
    [[0.666... 0.333...]]
    """
    _parameter_constraints: dict = {**NeighborsBase._parameter_constraints}
    _parameter_constraints.pop('radius')
    _parameter_constraints.update({'weights': [StrOptions({'uniform', 'distance'}), callable, None]})

    def __init__(self, n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
        super().__init__(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size, metric=metric, p=p, metric_params=metric_params, n_jobs=n_jobs)
        self.weights = weights

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or                 (n_samples, n_samples) if metric='precomputed'
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,) or                 (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        return self._fit(X, y)

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """
        check_is_fitted(self, '_fit_method')
        if self.weights == 'uniform':
            if self._fit_method == 'brute' and ArgKminClassMode.is_usable_for(X, self._fit_X, self.metric):
                probabilities = self.predict_proba(X)
                if self.outputs_2d_:
                    return np.stack([self.classes_[idx][np.argmax(probas, axis=1)] for idx, probas in enumerate(probabilities)], axis=1)
                return self.classes_[np.argmax(probabilities, axis=1)]
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)
        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]
        n_outputs = len(classes_)
        n_queries = _num_samples(X)
        weights = _get_weights(neigh_dist, self.weights)
        if weights is not None and _all_with_any_reduction_axis_1(weights, value=0):
            raise ValueError("All neighbors of some sample is getting zero weights. Please modify 'weights' to avoid this case if you are using a user-defined function.")
        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = _mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)
            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)
        if not self.outputs_2d_:
            y_pred = y_pred.ravel()
        return y_pred

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of n_outputs                 of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        check_is_fitted(self, '_fit_method')
        if self.weights == 'uniform':
            metric, metric_kwargs = _adjusted_metric(metric=self.metric, metric_kwargs=self.metric_params, p=self.p)
            if self._fit_method == 'brute' and ArgKminClassMode.is_usable_for(X, self._fit_X, metric) and (not self.outputs_2d_):
                if self.metric == 'precomputed':
                    X = _check_precomputed(X)
                else:
                    X = self._validate_data(X, accept_sparse='csr', reset=False, order='C')
                probabilities = ArgKminClassMode.compute(X, self._fit_X, k=self.n_neighbors, weights=self.weights, Y_labels=self._y, unique_Y_labels=self.classes_, metric=metric, metric_kwargs=metric_kwargs, strategy='parallel_on_X')
                return probabilities
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)
        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]
        n_queries = _num_samples(X)
        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)
        elif _all_with_any_reduction_axis_1(weights, value=0):
            raise ValueError("All neighbors of some sample is getting zero weights. Please modify 'weights' to avoid this case if you are using a user-defined function.")
        all_rows = np.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))
            for i, idx in enumerate(pred_labels.T):
                proba_k[all_rows, idx] += weights[:, i]
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            proba_k /= normalizer
            probabilities.append(proba_k)
        if not self.outputs_2d_:
            probabilities = probabilities[0]
        return probabilities

    def _more_tags(self):
        return {'multilabel': True}