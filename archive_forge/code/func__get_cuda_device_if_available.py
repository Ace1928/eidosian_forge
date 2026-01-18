from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target
def _get_cuda_device_if_available():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')