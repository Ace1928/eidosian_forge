from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
def _get_loss_function_for_train(params, estimator_type, train_pool):
    """
        estimator_type must be 'classifier', 'regressor', 'ranker' or None
        train_pool must be Pool
    """
    loss_function_param = params.get('loss_function')
    if loss_function_param is not None:
        return loss_function_param
    if estimator_type == 'classifier':
        if not isinstance(train_pool, Pool):
            raise CatBoostError('train_pool param must have Pool type')
        label = train_pool.get_label()
        if label is None:
            raise CatBoostError('loss function has not been specified and cannot be deduced')
        '\n            len(set) is faster than np.unique on Python lists:\n             https://bbengfort.github.io/observations/2017/05/02/python-unique-benchmark.html\n        '
        is_multiclass_task = len(set(label)) > 2 and 'target_border' not in params
        return 'MultiClass' if is_multiclass_task else 'Logloss'
    elif estimator_type == 'ranker':
        return 'YetiRank'
    else:
        return 'RMSE'