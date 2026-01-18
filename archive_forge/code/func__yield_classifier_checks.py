import pickle
import re
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial, wraps
from inspect import signature
from numbers import Integral, Real
import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from .. import config_context
from ..base import (
from ..datasets import (
from ..exceptions import DataConversionWarning, NotFittedError, SkipTestWarning
from ..feature_selection import SelectFromModel, SelectKBest
from ..linear_model import (
from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..metrics.pairwise import linear_kernel, pairwise_distances, rbf_kernel
from ..model_selection import ShuffleSplit, train_test_split
from ..model_selection._validation import _safe_split
from ..pipeline import make_pipeline
from ..preprocessing import StandardScaler, scale
from ..random_projection import BaseRandomProjection
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils._array_api import (
from ..utils._array_api import (
from ..utils._param_validation import (
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import check_is_fitted
from . import IS_PYPY, is_scalar_nan, shuffle
from ._param_validation import Interval
from ._tags import (
from ._testing import (
from .validation import _num_samples, has_fit_parameter
def _yield_classifier_checks(classifier):
    tags = _safe_tags(classifier)
    yield check_classifier_data_not_an_array
    yield check_classifiers_one_label
    yield check_classifiers_one_label_sample_weights
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    if tags['multioutput']:
        yield check_classifier_multioutput
    yield check_classifiers_train
    yield partial(check_classifiers_train, readonly_memmap=True)
    yield partial(check_classifiers_train, readonly_memmap=True, X_dtype='float32')
    yield check_classifiers_regression_target
    if tags['multilabel']:
        yield check_classifiers_multilabel_representation_invariance
        yield check_classifiers_multilabel_output_format_predict
        yield check_classifiers_multilabel_output_format_predict_proba
        yield check_classifiers_multilabel_output_format_decision_function
    if not tags['no_validation']:
        yield check_supervised_y_no_nan
        if not tags['multioutput_only']:
            yield check_supervised_y_2d
    if tags['requires_fit']:
        yield check_estimators_unfitted
    if 'class_weight' in classifier.get_params().keys():
        yield check_class_weight_classifiers
    yield check_non_transformer_estimators_n_iter
    yield check_decision_proba_consistency