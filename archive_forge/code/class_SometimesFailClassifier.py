from math import ceil
import numpy as np
import pytest
from scipy.stats import expon, norm, randint
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
from sklearn.model_selection._search_successive_halving import (
from sklearn.model_selection.tests.test_search import (
from sklearn.svm import SVC, LinearSVC
class SometimesFailClassifier(DummyClassifier):

    def __init__(self, strategy='stratified', random_state=None, constant=None, n_estimators=10, fail_fit=False, fail_predict=False, a=0):
        self.fail_fit = fail_fit
        self.fail_predict = fail_predict
        self.n_estimators = n_estimators
        self.a = a
        super().__init__(strategy=strategy, random_state=random_state, constant=constant)

    def fit(self, X, y):
        if self.fail_fit:
            raise Exception('fitting failed')
        return super().fit(X, y)

    def predict(self, X):
        if self.fail_predict:
            raise Exception('predict failed')
        return super().predict(X)