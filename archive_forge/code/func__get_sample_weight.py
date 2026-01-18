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
def _get_sample_weight(arg_names, args, kwargs):
    sample_weight_index = arg_names.index(_SAMPLE_WEIGHT)
    if len(args) > sample_weight_index:
        return args[sample_weight_index]
    if _SAMPLE_WEIGHT in kwargs:
        return kwargs[_SAMPLE_WEIGHT]
    return None