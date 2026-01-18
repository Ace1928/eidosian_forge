import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state, gen_batches, metadata_routing
from ..utils._param_validation import (
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import (
from ._cdnmf_fast import _update_cdnmf_fast
class _BaseNMF(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, ABC):
    """Base class for NMF and MiniBatchNMF."""
    __metadata_request__inverse_transform = {'W': metadata_routing.UNUSED}
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left'), None, StrOptions({'auto'}), Hidden(StrOptions({'warn'}))], 'init': [StrOptions({'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}), None], 'beta_loss': [StrOptions({'frobenius', 'kullback-leibler', 'itakura-saito'}), Real], 'tol': [Interval(Real, 0, None, closed='left')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state'], 'alpha_W': [Interval(Real, 0, None, closed='left')], 'alpha_H': [Interval(Real, 0, None, closed='left'), StrOptions({'same'})], 'l1_ratio': [Interval(Real, 0, 1, closed='both')], 'verbose': ['verbose']}

    def __init__(self, n_components='warn', *, init=None, beta_loss='frobenius', tol=0.0001, max_iter=200, random_state=None, alpha_W=0.0, alpha_H='same', l1_ratio=0.0, verbose=0):
        self.n_components = n_components
        self.init = init
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose

    def _check_params(self, X):
        self._n_components = self.n_components
        if self.n_components == 'warn':
            warnings.warn("The default value of `n_components` will change from `None` to `'auto'` in 1.6. Set the value of `n_components` to `None` explicitly to suppress the warning.", FutureWarning)
            self._n_components = None
        if self._n_components is None:
            self._n_components = X.shape[1]
        self._beta_loss = _beta_loss_to_float(self.beta_loss)

    def _check_w_h(self, X, W, H, update_H):
        """Check W and H, or initialize them."""
        n_samples, n_features = X.shape
        if self.init == 'custom' and update_H:
            _check_init(H, (self._n_components, n_features), 'NMF (input H)')
            _check_init(W, (n_samples, self._n_components), 'NMF (input W)')
            if self._n_components == 'auto':
                self._n_components = H.shape[0]
            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError('H and W should have the same dtype as X. Got H.dtype = {} and W.dtype = {}.'.format(H.dtype, W.dtype))
        elif not update_H:
            if W is not None:
                warnings.warn('When update_H=False, the provided initial W is not used.', RuntimeWarning)
            _check_init(H, (self._n_components, n_features), 'NMF (input H)')
            if self._n_components == 'auto':
                self._n_components = H.shape[0]
            if H.dtype != X.dtype:
                raise TypeError('H should have the same dtype as X. Got H.dtype = {}.'.format(H.dtype))
            if self.solver == 'mu':
                avg = np.sqrt(X.mean() / self._n_components)
                W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
            else:
                W = np.zeros((n_samples, self._n_components), dtype=X.dtype)
        else:
            if W is not None or H is not None:
                warnings.warn("When init!='custom', provided W or H are ignored. Set  init='custom' to use them as initialization.", RuntimeWarning)
            if self._n_components == 'auto':
                self._n_components = X.shape[1]
            W, H = _initialize_nmf(X, self._n_components, init=self.init, random_state=self.random_state)
        return (W, H)

    def _compute_regularization(self, X):
        """Compute scaled regularization terms."""
        n_samples, n_features = X.shape
        alpha_W = self.alpha_W
        alpha_H = self.alpha_W if self.alpha_H == 'same' else self.alpha_H
        l1_reg_W = n_features * alpha_W * self.l1_ratio
        l1_reg_H = n_samples * alpha_H * self.l1_ratio
        l2_reg_W = n_features * alpha_W * (1.0 - self.l1_ratio)
        l2_reg_H = n_samples * alpha_H * (1.0 - self.l1_ratio)
        return (l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H)

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        **params : kwargs
            Parameters (keyword arguments) and values passed to
            the fit_transform instance.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.fit_transform(X, **params)
        return self

    def inverse_transform(self, Xt=None, W=None):
        """Transform data back to its original space.

        .. versionadded:: 0.18

        Parameters
        ----------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.

        W : deprecated
            Use `Xt` instead.

            .. deprecated:: 1.3

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Returns a data matrix of the original shape.
        """
        if Xt is None and W is None:
            raise TypeError('Missing required positional argument: Xt')
        if W is not None and Xt is not None:
            raise ValueError('Please provide only `Xt`, and not `W`.')
        if W is not None:
            warnings.warn('Input argument `W` was renamed to `Xt` in v1.3 and will be removed in v1.5.', FutureWarning)
            Xt = W
        check_is_fitted(self)
        return Xt @ self.components_

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.components_.shape[0]

    def _more_tags(self):
        return {'requires_positive_X': True, 'preserves_dtype': [np.float64, np.float32]}