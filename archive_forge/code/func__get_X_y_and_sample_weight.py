import collections
import inspect
import logging
import pkgutil
import platform
import warnings
from copy import deepcopy
from importlib import import_module
from numbers import Number
from operator import itemgetter
import numpy as np
from packaging.version import Version
from mlflow import MlflowClient
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.time import get_current_time_millis
def _get_X_y_and_sample_weight(fit_func, fit_args, fit_kwargs):
    """
    Get a tuple of (X, y, sample_weight) in the following steps.

    1. Extract X and y from fit_args and fit_kwargs.
    2. If the sample_weight argument exists in fit_func,
       extract it from fit_args or fit_kwargs and return (X, y, sample_weight),
       otherwise return (X, y)

    Args:
        fit_func: A fit function object.
        fit_args: Positional arguments given to fit_func.
        fit_kwargs: Keyword arguments given to fit_func.

    Returns:
        A tuple of either (X, y, sample_weight), where `y` and `sample_weight` may be
        `None` if the specified `fit_args` and `fit_kwargs` do not specify labels or
        a sample weighting. Copies of `X` and `y` are made in order to avoid mutation
        of the dataset during training.
    """

    def _get_Xy(args, kwargs, X_var_name, y_var_name):
        if len(args) >= 2:
            return args[:2]
        if len(args) == 1:
            return (args[0], kwargs.get(y_var_name))
        return (kwargs[X_var_name], kwargs.get(y_var_name))

    def _get_sample_weight(arg_names, args, kwargs):
        sample_weight_index = arg_names.index(_SAMPLE_WEIGHT)
        if len(args) > sample_weight_index:
            return args[sample_weight_index]
        if _SAMPLE_WEIGHT in kwargs:
            return kwargs[_SAMPLE_WEIGHT]
        return None
    fit_arg_names = _get_arg_names(fit_func)
    X_var_name, y_var_name = fit_arg_names[:2]
    X, y = _get_Xy(fit_args, fit_kwargs, X_var_name, y_var_name)
    if X is not None:
        X = deepcopy(X)
    if y is not None:
        y = deepcopy(y)
    sample_weight = _get_sample_weight(fit_arg_names, fit_args, fit_kwargs) if _SAMPLE_WEIGHT in fit_arg_names else None
    return (X, y, sample_weight)