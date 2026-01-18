from abc import ABCMeta, abstractmethod
from copy import deepcopy
from numbers import Integral
import numpy as np
import scipy.sparse as sparse
from ..base import (
from ..exceptions import NotFittedError
from ..linear_model import LogisticRegression, RidgeCV
from ..model_selection import check_cv, cross_val_predict
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._param_validation import HasMethods, StrOptions
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
@staticmethod
def _method_name(name, estimator, method):
    if estimator == 'drop':
        return None
    if method == 'auto':
        method = ['predict_proba', 'decision_function', 'predict']
    try:
        method_name = _check_response_method(estimator, method).__name__
    except AttributeError as e:
        raise ValueError(f'Underlying estimator {name} does not implement the method {method}.') from e
    return method_name