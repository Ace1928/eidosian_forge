import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect
class ARGType(object):
    """Base class for parameter specifications."""
    pass