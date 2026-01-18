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
def _get_meta_estimators_for_autologging():
    """
    Returns:
        A list of meta estimator class definitions
        (e.g., `sklearn.model_selection.GridSearchCV`) that should be included
        when patching training functions for autologging
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    return [GridSearchCV, RandomizedSearchCV, Pipeline]