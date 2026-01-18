from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from ..base import (
from ..exceptions import NotFittedError
from ..metrics.pairwise import pairwise_kernels
from ..preprocessing import KernelCenterer
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import _randomized_eigsh, svd_flip
from ..utils.validation import (
def _fit_transform(self, K):
    """Fit's using kernel K"""
    K = self._centerer.fit_transform(K)
    if self.n_components is None:
        n_components = K.shape[0]
    else:
        n_components = min(K.shape[0], self.n_components)
    if self.eigen_solver == 'auto':
        if K.shape[0] > 200 and n_components < 10:
            eigen_solver = 'arpack'
        else:
            eigen_solver = 'dense'
    else:
        eigen_solver = self.eigen_solver
    if eigen_solver == 'dense':
        self.eigenvalues_, self.eigenvectors_ = eigh(K, subset_by_index=(K.shape[0] - n_components, K.shape[0] - 1))
    elif eigen_solver == 'arpack':
        v0 = _init_arpack_v0(K.shape[0], self.random_state)
        self.eigenvalues_, self.eigenvectors_ = eigsh(K, n_components, which='LA', tol=self.tol, maxiter=self.max_iter, v0=v0)
    elif eigen_solver == 'randomized':
        self.eigenvalues_, self.eigenvectors_ = _randomized_eigsh(K, n_components=n_components, n_iter=self.iterated_power, random_state=self.random_state, selection='module')
    self.eigenvalues_ = _check_psd_eigenvalues(self.eigenvalues_, enable_warnings=False)
    self.eigenvectors_, _ = svd_flip(self.eigenvectors_, np.zeros_like(self.eigenvectors_).T)
    indices = self.eigenvalues_.argsort()[::-1]
    self.eigenvalues_ = self.eigenvalues_[indices]
    self.eigenvectors_ = self.eigenvectors_[:, indices]
    if self.remove_zero_eig or self.n_components is None:
        self.eigenvectors_ = self.eigenvectors_[:, self.eigenvalues_ > 0]
        self.eigenvalues_ = self.eigenvalues_[self.eigenvalues_ > 0]
    return K