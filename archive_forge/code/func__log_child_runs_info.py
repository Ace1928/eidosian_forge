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
def _log_child_runs_info(max_tuning_runs, total_runs):
    rest = total_runs - max_tuning_runs
    if max_tuning_runs == 0:
        logging_phrase = 'no runs'
    elif max_tuning_runs == 1:
        logging_phrase = 'the best run'
    else:
        logging_phrase = f'the {max_tuning_runs} best runs'
    if rest <= 0:
        omitting_phrase = 'no runs'
    elif rest == 1:
        omitting_phrase = 'one run'
    else:
        omitting_phrase = f'{rest} runs'
    _logger.info('Logging %s, %s will be omitted.', logging_phrase, omitting_phrase)