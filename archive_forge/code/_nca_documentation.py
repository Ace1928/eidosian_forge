import sys
import time
from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.optimize import minimize
from ..base import (
from ..decomposition import PCA
from ..exceptions import ConvergenceWarning
from ..metrics import pairwise_distances
from ..preprocessing import LabelEncoder
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import softmax
from ..utils.multiclass import check_classification_targets
from ..utils.random import check_random_state
from ..utils.validation import check_array, check_is_fitted
Compute the loss and the loss gradient w.r.t. `transformation`.

        Parameters
        ----------
        transformation : ndarray of shape (n_components * n_features,)
            The raveled linear transformation on which to compute loss and
            evaluate gradient.

        X : ndarray of shape (n_samples, n_features)
            The training samples.

        same_class_mask : ndarray of shape (n_samples, n_samples)
            A mask where `mask[i, j] == 1` if `X[i]` and `X[j]` belong
            to the same class, and `0` otherwise.

        Returns
        -------
        loss : float
            The loss computed for the given transformation.

        gradient : ndarray of shape (n_components * n_features,)
            The new (flattened) gradient of the loss.
        