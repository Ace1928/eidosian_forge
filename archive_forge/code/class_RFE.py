import re
from pprint import PrettyPrinter
import numpy as np
from sklearn.utils._pprint import _EstimatorPrettyPrinter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import config_context
class RFE(BaseEstimator):

    def __init__(self, estimator, n_features_to_select=None, step=1, verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose