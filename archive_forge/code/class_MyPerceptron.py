import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal
from sklearn.utils.fixes import CSR_CONTAINERS
class MyPerceptron:

    def __init__(self, n_iter=1):
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        for t in range(self.n_iter):
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i]
                    self.b += y[i]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))