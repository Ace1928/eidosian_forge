import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .base import (
from .exceptions import DataDimensionalityWarning
from .utils import check_random_state
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.extmath import safe_sparse_dot
from .utils.random import sample_without_replacement
from .utils.validation import check_array, check_is_fitted
class GaussianRandomProjection(BaseRandomProjection):
    """Reduce dimensionality through Gaussian random projection.

    The components of the random matrix are drawn from N(0, 1 / n_components).

    Read more in the :ref:`User Guide <gaussian_random_matrix>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    eps : float, default=0.1
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when `n_components` is set to
        'auto'. The value should be strictly positive.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    compute_inverse_components : bool, default=False
        Learn the inverse transform by computing the pseudo-inverse of the
        components during fit. Note that computing the pseudo-inverse does not
        scale well to large matrices.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the
        projection matrix at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    n_components_ : int
        Concrete number of components computed when n_components="auto".

    components_ : ndarray of shape (n_components, n_features)
        Random matrix used for the projection.

    inverse_components_ : ndarray of shape (n_features, n_components)
        Pseudo-inverse of the components, only computed if
        `compute_inverse_components` is True.

        .. versionadded:: 1.1

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SparseRandomProjection : Reduce dimensionality through sparse
        random projection.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.random_projection import GaussianRandomProjection
    >>> rng = np.random.RandomState(42)
    >>> X = rng.rand(25, 3000)
    >>> transformer = GaussianRandomProjection(random_state=rng)
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (25, 2759)
    """

    def __init__(self, n_components='auto', *, eps=0.1, compute_inverse_components=False, random_state=None):
        super().__init__(n_components=n_components, eps=eps, compute_inverse_components=compute_inverse_components, random_state=random_state)

    def _make_random_matrix(self, n_components, n_features):
        """Generate the random projection matrix.

        Parameters
        ----------
        n_components : int,
            Dimensionality of the target projection space.

        n_features : int,
            Dimensionality of the original source space.

        Returns
        -------
        components : ndarray of shape (n_components, n_features)
            The generated random matrix.
        """
        random_state = check_random_state(self.random_state)
        return _gaussian_random_matrix(n_components, n_features, random_state=random_state)

    def transform(self, X):
        """Project the data by using matrix product with the random matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected array.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=['csr', 'csc'], reset=False, dtype=[np.float64, np.float32])
        return X @ self.components_.T