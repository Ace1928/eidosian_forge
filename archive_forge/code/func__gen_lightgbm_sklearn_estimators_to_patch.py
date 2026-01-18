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
def _gen_lightgbm_sklearn_estimators_to_patch():
    import lightgbm as lgb
    import mlflow.lightgbm
    all_classes = inspect.getmembers(lgb.sklearn, inspect.isclass)
    base_class = lgb.sklearn._LGBMModelBase
    sklearn_estimators = []
    for _, class_object in all_classes:
        package_name = class_object.__module__.split('.')[0]
        if package_name == mlflow.lightgbm.FLAVOR_NAME and issubclass(class_object, base_class) and (class_object != base_class):
            sklearn_estimators.append(class_object)
    return sklearn_estimators