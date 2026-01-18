from . import utils
from scipy import sparse
from sklearn import decomposition
from sklearn import random_projection
import numpy as np
import pandas as pd
import sklearn.base
import warnings
class AutomaticDimensionSVD(decomposition.TruncatedSVD):
    """Truncated SVD with automatic dimensionality selected by Johnson-Lindenstrauss."""

    def __init__(self, n_components='auto', eps=0.3, algorithm='randomized', n_iter=5, random_state=None, tol=0.0):
        self.eps = eps
        if n_components == 'auto':
            n_components = -1
        super().__init__(n_components=n_components, algorithm=algorithm, n_iter=n_iter, random_state=random_state, tol=tol)

    def fit(self, X):
        if self.n_components == -1:
            super().set_params(n_components=random_projection.johnson_lindenstrauss_min_dim(n_samples=X.shape[0], eps=self.eps))
        try:
            return super().fit(X)
        except ValueError:
            if self.n_components >= X.shape[1]:
                raise RuntimeError('eps={} and n_samples={} lead to a target dimension of {} which is larger than the original space with n_features={}'.format(self.eps, X.shape[0], self.n_components, X.shape[1]))
            else:
                raise