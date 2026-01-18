import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect
def _is_transformer(estimator):
    return hasattr(estimator, 'fit_transform')