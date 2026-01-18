import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.fixes import laplacian as csgraph_laplacian
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
class LabelPropagation(BaseLabelPropagation):
    """Label Propagation classifier.

    Read more in the :ref:`User Guide <label_propagation>`.

    Parameters
    ----------
    kernel : {'knn', 'rbf'} or callable, default='rbf'
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape (n_samples, n_features),
        and return a (n_samples, n_samples) shaped weight matrix.

    gamma : float, default=20
        Parameter for rbf kernel.

    n_neighbors : int, default=7
        Parameter for knn kernel which need to be strictly positive.

    max_iter : int, default=1000
        Change maximum number of iterations allowed.

    tol : float, 1e-3
        Convergence tolerance: threshold to consider the system at steady
        state.

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    X_ : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input array.

    classes_ : ndarray of shape (n_classes,)
        The distinct labels used in classifying instances.

    label_distributions_ : ndarray of shape (n_samples, n_classes)
        Categorical distribution for each item.

    transduction_ : ndarray of shape (n_samples)
        Label assigned to each item during :term:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    LabelSpreading : Alternate label propagation strategy more robust to noise.

    References
    ----------
    Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data
    with label propagation. Technical Report CMU-CALD-02-107, Carnegie Mellon
    University, 2002 http://pages.cs.wisc.edu/~jerryzhu/pub/CMU-CALD-02-107.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import LabelPropagation
    >>> label_prop_model = LabelPropagation()
    >>> iris = datasets.load_iris()
    >>> rng = np.random.RandomState(42)
    >>> random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    >>> labels = np.copy(iris.target)
    >>> labels[random_unlabeled_points] = -1
    >>> label_prop_model.fit(iris.data, labels)
    LabelPropagation(...)
    """
    _variant = 'propagation'
    _parameter_constraints: dict = {**BaseLabelPropagation._parameter_constraints}
    _parameter_constraints.pop('alpha')

    def __init__(self, kernel='rbf', *, gamma=20, n_neighbors=7, max_iter=1000, tol=0.001, n_jobs=None):
        super().__init__(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol, n_jobs=n_jobs, alpha=None)

    def _build_graph(self):
        """Matrix representing a fully connected graph between each sample

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired).
        """
        if self.kernel == 'knn':
            self.nn_fit = None
        affinity_matrix = self._get_kernel(self.X_)
        normalizer = affinity_matrix.sum(axis=0)
        if sparse.issparse(affinity_matrix):
            affinity_matrix.data /= np.diag(np.array(normalizer))
        else:
            affinity_matrix /= normalizer[:, np.newaxis]
        return affinity_matrix

    def fit(self, X, y):
        """Fit a semi-supervised label propagation model to X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target class values with unlabeled points marked as -1.
            All unlabeled samples will be transductively assigned labels
            internally, which are stored in `transduction_`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return super().fit(X, y)