import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from ..base import (
from ..exceptions import DataConversionWarning
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import BaseEnsemble, _partition_estimators
@staticmethod
def _get_oob_predictions(tree, X):
    """Compute the OOB predictions for an individual tree.

        Parameters
        ----------
        tree : DecisionTreeRegressor object
            A single decision tree regressor.
        X : ndarray of shape (n_samples, n_features)
            The OOB samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, 1, n_outputs)
            The OOB associated predictions.
        """
    y_pred = tree.predict(X, check_input=False)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis, np.newaxis]
    else:
        y_pred = y_pred[:, np.newaxis, :]
    return y_pred