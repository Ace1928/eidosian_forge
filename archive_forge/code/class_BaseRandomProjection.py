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
class BaseRandomProjection(TransformerMixin, BaseEstimator, ClassNamePrefixFeaturesOutMixin, metaclass=ABCMeta):
    """Base class for random projections.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left'), StrOptions({'auto'})], 'eps': [Interval(Real, 0, None, closed='neither')], 'compute_inverse_components': ['boolean'], 'random_state': ['random_state']}

    @abstractmethod
    def __init__(self, n_components='auto', *, eps=0.1, compute_inverse_components=False, random_state=None):
        self.n_components = n_components
        self.eps = eps
        self.compute_inverse_components = compute_inverse_components
        self.random_state = random_state

    @abstractmethod
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
        components : {ndarray, sparse matrix} of shape (n_components, n_features)
            The generated random matrix. Sparse matrix will be of CSR format.

        """

    def _compute_inverse_components(self):
        """Compute the pseudo-inverse of the (densified) components."""
        components = self.components_
        if sp.issparse(components):
            components = components.toarray()
        return linalg.pinv(components, check_finite=False)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Generate a sparse random projection matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training set: only the shape is used to find optimal random
            matrix dimensions based on the theory referenced in the
            afore mentioned papers.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            BaseRandomProjection class instance.
        """
        X = self._validate_data(X, accept_sparse=['csr', 'csc'], dtype=[np.float64, np.float32])
        n_samples, n_features = X.shape
        if self.n_components == 'auto':
            self.n_components_ = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)
            if self.n_components_ <= 0:
                raise ValueError('eps=%f and n_samples=%d lead to a target dimension of %d which is invalid' % (self.eps, n_samples, self.n_components_))
            elif self.n_components_ > n_features:
                raise ValueError('eps=%f and n_samples=%d lead to a target dimension of %d which is larger than the original space with n_features=%d' % (self.eps, n_samples, self.n_components_, n_features))
        else:
            if self.n_components > n_features:
                warnings.warn('The number of components is higher than the number of features: n_features < n_components (%s < %s).The dimensionality of the problem will not be reduced.' % (n_features, self.n_components), DataDimensionalityWarning)
            self.n_components_ = self.n_components
        self.components_ = self._make_random_matrix(self.n_components_, n_features).astype(X.dtype, copy=False)
        if self.compute_inverse_components:
            self.inverse_components_ = self._compute_inverse_components()
        self._n_features_out = self.n_components
        return self

    def inverse_transform(self, X):
        """Project data back to its original space.

        Returns an array X_original whose transform would be X. Note that even
        if X is sparse, X_original is dense: this may use a lot of RAM.

        If `compute_inverse_components` is False, the inverse of the components is
        computed during each call to `inverse_transform` which can be costly.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_components)
            Data to be transformed back.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=[np.float64, np.float32], accept_sparse=('csr', 'csc'))
        if self.compute_inverse_components:
            return X @ self.inverse_components_.T
        inverse_components = self._compute_inverse_components()
        return X @ inverse_components.T

    def _more_tags(self):
        return {'preserves_dtype': [np.float64, np.float32]}