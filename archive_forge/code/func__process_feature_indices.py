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
def _process_feature_indices(feature_indices, pool, params, param_name):
    if param_name not in params:
        return feature_indices
    if param_name == 'cat_features':
        feature_type_name = 'categorical'
    elif param_name == 'text_features':
        feature_type_name = 'text'
    elif param_name == 'embedding_features':
        feature_type_name = 'embedding'
    else:
        raise CatBoostError('Unknown params_name=' + param_name)
    if isinstance(pool, Pool):
        feature_indices_from_params = _get_features_indices(params[param_name], pool.get_feature_names())
        if param_name == 'cat_features':
            feature_indices_from_pool = pool.get_cat_feature_indices()
        elif param_name == 'text_features':
            feature_indices_from_pool = pool.get_text_feature_indices()
        else:
            feature_indices_from_pool = pool.get_embedding_feature_indices()
        if set(feature_indices_from_pool) != set(feature_indices_from_params):
            raise CatBoostError(feature_type_name + ' features indices in the model are set to ' + str(feature_indices_from_params) + ' and train dataset ' + feature_type_name + ' features indices are set to ' + str(feature_indices_from_pool))
    elif isinstance(pool, FeaturesData):
        raise CatBoostError('Categorical features are set in the model. It is not allowed to use FeaturesData type for training dataset.')
    else:
        if feature_indices is not None and set(feature_indices) != set(params[param_name]):
            raise CatBoostError(feature_type_name + ' features in the model are set to ' + str(params[param_name]) + '. ' + feature_type_name + ' features passed to fit function are set to ' + str(feature_indices))
        feature_indices = params[param_name]
    del params[param_name]
    return feature_indices