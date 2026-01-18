import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect
def _is_selector(estimator):
    selector_attributes = ['get_support', 'transform', 'inverse_transform', 'fit_transform']
    return all((hasattr(estimator, attr) for attr in selector_attributes))