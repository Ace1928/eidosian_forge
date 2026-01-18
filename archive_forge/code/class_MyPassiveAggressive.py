import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
class MyPassiveAggressive(ClassifierMixin):

    def __init__(self, C=1.0, epsilon=0.01, loss='hinge', fit_intercept=True, n_iter=1, random_state=None):
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        for t in range(self.n_iter):
            for i in range(n_samples):
                p = self.project(X[i])
                if self.loss in ('hinge', 'squared_hinge'):
                    loss = max(1 - y[i] * p, 0)
                else:
                    loss = max(np.abs(p - y[i]) - self.epsilon, 0)
                sqnorm = np.dot(X[i], X[i])
                if self.loss in ('hinge', 'epsilon_insensitive'):
                    step = min(self.C, loss / sqnorm)
                elif self.loss in ('squared_hinge', 'squared_epsilon_insensitive'):
                    step = loss / (sqnorm + 1.0 / (2 * self.C))
                if self.loss in ('hinge', 'squared_hinge'):
                    step *= y[i]
                else:
                    step *= np.sign(y[i] - p)
                self.w += step * X[i]
                if self.fit_intercept:
                    self.b += step

    def project(self, X):
        return np.dot(X, self.w) + self.b