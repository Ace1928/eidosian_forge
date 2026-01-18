import copy
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import (
import numpy as np
from scipy.special import softmax
from ._typing import ArrayLike, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import SKLEARN_INSTALLED, XGBClassifierBase, XGBModelBase, XGBRegressorBase
from .config import config_context
from .core import (
from .data import _is_cudf_df, _is_cudf_ser, _is_cupy_array, _is_pandas_df
from .training import train
def _metric_decorator(func: Callable) -> Metric:
    """Decorate a metric function from sklearn.

    Converts an metric function that uses the typical sklearn metric signature so that
    it is compatible with :py:func:`train`

    """

    def inner(y_score: np.ndarray, dmatrix: DMatrix) -> Tuple[str, float]:
        y_true = dmatrix.get_label()
        weight = dmatrix.get_weight()
        if weight.size == 0:
            return (func.__name__, func(y_true, y_score))
        return (func.__name__, func(y_true, y_score, sample_weight=weight))
    return inner