import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
def _enforce_prediction_order(classes, predictions, n_classes, method):
    """Ensure that prediction arrays have correct column order

    When doing cross-validation, if one or more classes are
    not present in the subset of data used for training,
    then the output prediction array might not have the same
    columns as other folds. Use the list of class names
    (assumed to be ints) to enforce the correct column order.

    Note that `classes` is the list of classes in this fold
    (a subset of the classes in the full training set)
    and `n_classes` is the number of classes in the full training set.
    """
    if n_classes != len(classes):
        recommendation = 'To fix this, use a cross-validation technique resulting in properly stratified folds'
        warnings.warn('Number of classes in training fold ({}) does not match total number of classes ({}). Results may not be appropriate for your use case. {}'.format(len(classes), n_classes, recommendation), RuntimeWarning)
        if method == 'decision_function':
            if predictions.ndim == 2 and predictions.shape[1] != len(classes):
                raise ValueError('Output shape {} of {} does not match number of classes ({}) in fold. Irregular decision_function outputs are not currently supported by cross_val_predict'.format(predictions.shape, method, len(classes)))
            if len(classes) <= 2:
                raise ValueError('Only {} class/es in training fold, but {} in overall dataset. This is not supported for decision_function with imbalanced folds. {}'.format(len(classes), n_classes, recommendation))
        float_min = np.finfo(predictions.dtype).min
        default_values = {'decision_function': float_min, 'predict_log_proba': float_min, 'predict_proba': 0}
        predictions_for_all_classes = np.full((_num_samples(predictions), n_classes), default_values[method], dtype=predictions.dtype)
        predictions_for_all_classes[:, classes] = predictions
        predictions = predictions_for_all_classes
    return predictions