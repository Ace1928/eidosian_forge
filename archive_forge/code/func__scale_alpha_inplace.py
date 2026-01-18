import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def _scale_alpha_inplace(estimator, n_samples):
    """Rescale the parameter alpha from when the estimator is evoked with
    normalize set to True as if it were evoked in a Pipeline with normalize set
    to False and with a StandardScaler.
    """
    if 'alpha' not in estimator.get_params() and 'alphas' not in estimator.get_params():
        return
    if isinstance(estimator, (RidgeCV, RidgeClassifierCV)):
        alphas = np.asarray(estimator.alphas) * n_samples
        return estimator.set_params(alphas=alphas)
    if isinstance(estimator, (Lasso, LassoLars, MultiTaskLasso)):
        alpha = estimator.alpha * np.sqrt(n_samples)
    if isinstance(estimator, (Ridge, RidgeClassifier)):
        alpha = estimator.alpha * n_samples
    if isinstance(estimator, (ElasticNet, MultiTaskElasticNet)):
        if estimator.l1_ratio == 1:
            alpha = estimator.alpha * np.sqrt(n_samples)
        elif estimator.l1_ratio == 0:
            alpha = estimator.alpha * n_samples
        else:
            raise NotImplementedError
    estimator.set_params(alpha=alpha)