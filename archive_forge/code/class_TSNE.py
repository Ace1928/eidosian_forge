from numbers import Integral, Real
from time import time
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
from ..base import (
from ..decomposition import PCA
from ..metrics.pairwise import _VALID_METRICS, pairwise_distances
from ..neighbors import NearestNeighbors
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import _num_samples, check_non_negative
from . import _barnes_hut_tsne, _utils  # type: ignore
class TSNE(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """T-distributed Stochastic Neighbor Embedding.

    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].

    Read more in the :ref:`User Guide <t_sne>`.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results. The perplexity must be less than the number
        of samples.

    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.

    learning_rate : float or "auto", default="auto"
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
        Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE,
        etc.) use a definition of learning_rate that is 4 times smaller than
        ours. So our learning_rate=200 corresponds to learning_rate=800 in
        those other implementations. The 'auto' option sets the learning_rate
        to `max(N / early_exaggeration / 4, 50)` where N is the sample size,
        following [4] and [5].

        .. versionchanged:: 1.2
           The default value changed to `"auto"`.

    n_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.

        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 1.1

    init : {"random", "pca"} or ndarray of shape (n_samples, n_components),             default="pca"
        Initialization of embedding.
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

        .. versionchanged:: 1.2
           The default value changed to `"pca"`.

    verbose : int, default=0
        Verbosity level.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term:`Glossary <random_state>`.

    method : {'barnes_hut', 'exact'}, default='barnes_hut'
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.

    angle : float, default=0.5
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.22

    Attributes
    ----------
    embedding_ : array-like of shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    learning_rate_ : float
        Effective learning rate.

        .. versionadded:: 1.2

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis that is a linear
        dimensionality reduction method.
    sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
        kernels and PCA.
    MDS : Manifold learning using multidimensional scaling.
    Isomap : Manifold learning based on Isometric Mapping.
    LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
    SpectralEmbedding : Spectral embedding for non-linear dimensionality.

    Notes
    -----
    For an example of using :class:`~sklearn.manifold.TSNE` in combination with
    :class:`~sklearn.neighbors.KNeighborsTransformer` see
    :ref:`sphx_glr_auto_examples_neighbors_approximate_nearest_neighbors.py`.

    References
    ----------

    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/

    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf

    [4] Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J.,
        & Snyder-Cappione, J. E. (2019). Automated optimized parameters for
        T-distributed stochastic neighbor embedding improve visualization
        and analysis of large datasets. Nature Communications, 10(1), 1-12.

    [5] Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell
        transcriptomics. Nature Communications, 10(1), 1-14.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2, learning_rate='auto',
    ...                   init='random', perplexity=3).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'perplexity': [Interval(Real, 0, None, closed='neither')], 'early_exaggeration': [Interval(Real, 1, None, closed='left')], 'learning_rate': [StrOptions({'auto'}), Interval(Real, 0, None, closed='neither')], 'n_iter': [Interval(Integral, 250, None, closed='left')], 'n_iter_without_progress': [Interval(Integral, -1, None, closed='left')], 'min_grad_norm': [Interval(Real, 0, None, closed='left')], 'metric': [StrOptions(set(_VALID_METRICS) | {'precomputed'}), callable], 'metric_params': [dict, None], 'init': [StrOptions({'pca', 'random'}), np.ndarray], 'verbose': ['verbose'], 'random_state': ['random_state'], 'method': [StrOptions({'barnes_hut', 'exact'})], 'angle': [Interval(Real, 0, 1, closed='both')], 'n_jobs': [None, Integral]}
    _EXPLORATION_N_ITER = 250
    _N_ITER_CHECK = 50

    def __init__(self, n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', metric_params=None, init='pca', verbose=0, random_state=None, method='barnes_hut', angle=0.5, n_jobs=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs

    def _check_params_vs_input(self, X):
        if self.perplexity >= X.shape[0]:
            raise ValueError('perplexity must be less than n_samples')

    def _fit(self, X, skip_num_points=0):
        """Private function to fit the model using X as training data."""
        if isinstance(self.init, str) and self.init == 'pca' and issparse(X):
            raise TypeError('PCA initialization is currently not supported with the sparse input matrix. Use init="random" instead.')
        if self.learning_rate == 'auto':
            self.learning_rate_ = X.shape[0] / self.early_exaggeration / 4
            self.learning_rate_ = np.maximum(self.learning_rate_, 50)
        else:
            self.learning_rate_ = self.learning_rate
        if self.method == 'barnes_hut':
            X = self._validate_data(X, accept_sparse=['csr'], ensure_min_samples=2, dtype=[np.float32, np.float64])
        else:
            X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'], dtype=[np.float32, np.float64])
        if self.metric == 'precomputed':
            if isinstance(self.init, str) and self.init == 'pca':
                raise ValueError('The parameter init="pca" cannot be used with metric="precomputed".')
            if X.shape[0] != X.shape[1]:
                raise ValueError('X should be a square distance matrix')
            check_non_negative(X, "TSNE.fit(). With metric='precomputed', X should contain positive distances.")
            if self.method == 'exact' and issparse(X):
                raise TypeError('TSNE with method="exact" does not accept sparse precomputed distance matrix. Use method="barnes_hut" or provide the dense distance matrix.')
        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)
        n_samples = X.shape[0]
        neighbors_nn = None
        if self.method == 'exact':
            if self.metric == 'precomputed':
                distances = X
            else:
                if self.verbose:
                    print('[t-SNE] Computing pairwise distances...')
                if self.metric == 'euclidean':
                    distances = pairwise_distances(X, metric=self.metric, squared=True)
                else:
                    metric_params_ = self.metric_params or {}
                    distances = pairwise_distances(X, metric=self.metric, n_jobs=self.n_jobs, **metric_params_)
            if np.any(distances < 0):
                raise ValueError('All distances should be positive, the metric given is not correct')
            if self.metric != 'euclidean':
                distances **= 2
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), 'All probabilities should be finite'
            assert np.all(P >= 0), 'All probabilities should be non-negative'
            assert np.all(P <= 1), 'All probabilities should be less or then equal to one'
        else:
            n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))
            if self.verbose:
                print('[t-SNE] Computing {} nearest neighbors...'.format(n_neighbors))
            knn = NearestNeighbors(algorithm='auto', n_jobs=self.n_jobs, n_neighbors=n_neighbors, metric=self.metric, metric_params=self.metric_params)
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print('[t-SNE] Indexed {} samples in {:.3f}s...'.format(n_samples, duration))
            t0 = time()
            distances_nn = knn.kneighbors_graph(mode='distance')
            duration = time() - t0
            if self.verbose:
                print('[t-SNE] Computed neighbors for {} samples in {:.3f}s...'.format(n_samples, duration))
            del knn
            distances_nn.data **= 2
            P = _joint_probabilities_nn(distances_nn, self.perplexity, self.verbose)
        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized', random_state=random_state)
            pca.set_output(transform='default')
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 0.0001
        elif self.init == 'random':
            X_embedded = 0.0001 * random_state.standard_normal(size=(n_samples, self.n_components)).astype(np.float32)
        degrees_of_freedom = max(self.n_components - 1, 1)
        return self._tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded, neighbors=neighbors_nn, skip_num_points=skip_num_points)

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded, neighbors=None, skip_num_points=0):
        """Runs t-SNE."""
        params = X_embedded.ravel()
        opt_args = {'it': 0, 'n_iter_check': self._N_ITER_CHECK, 'min_grad_norm': self.min_grad_norm, 'learning_rate': self.learning_rate_, 'verbose': self.verbose, 'kwargs': dict(skip_num_points=skip_num_points), 'args': [P, degrees_of_freedom, n_samples, self.n_components], 'n_iter_without_progress': self._EXPLORATION_N_ITER, 'n_iter': self._EXPLORATION_N_ITER, 'momentum': 0.5}
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            opt_args['kwargs']['verbose'] = self.verbose
            opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
        else:
            obj_func = _kl_divergence
        P *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
        if self.verbose:
            print('[t-SNE] KL divergence after %d iterations with early exaggeration: %f' % (it + 1, kl_divergence))
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
        self.n_iter_ = it
        if self.verbose:
            print('[t-SNE] KL divergence after %d iterations: %f' % (it + 1, kl_divergence))
        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence
        return X_embedded

    @_fit_context(prefer_skip_nested_validation=False)
    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or             (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self._check_params_vs_input(X)
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or             (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X)
        return self

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.embedding_.shape[1]

    def _more_tags(self):
        return {'pairwise': self.metric == 'precomputed'}