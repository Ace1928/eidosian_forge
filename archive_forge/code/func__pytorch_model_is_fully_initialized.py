from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target
def _pytorch_model_is_fully_initialized(clf: BaseEstimator):
    if all([hasattr(clf, 'network'), hasattr(clf, 'loss_function'), hasattr(clf, 'optimizer'), hasattr(clf, 'data_loader'), hasattr(clf, 'train_dset_len'), hasattr(clf, 'device')]):
        return True
    else:
        return False