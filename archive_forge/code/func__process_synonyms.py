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
def _process_synonyms(params):
    if 'objective' in params:
        params['loss_function'] = params['objective']
        del params['objective']
    if 'scale_pos_weight' in params:
        if 'loss_function' in params and params['loss_function'] != 'Logloss':
            raise CatBoostError('scale_pos_weight is supported only for binary classification Logloss loss')
        if 'class_weights' in params or 'auto_class_weights' in params:
            raise CatBoostError('only one of the parameters scale_pos_weight, class_weights, auto_class_weights should be initialized.')
        params['class_weights'] = [1.0, params['scale_pos_weight']]
        del params['scale_pos_weight']
    if 'class_weights' in params and isinstance(params['class_weights'], (dict, OrderedDict)):
        class_weights_dict = params['class_weights']
        class_weights_list = []
        if 'class_names' in params and params['class_names'] is not None:
            if len(class_weights_dict) != len(params['class_names']):
                raise CatBoostError('Number of classes in class_names and class_weights differ')
            for class_label in params['class_names']:
                if class_label not in class_weights_dict:
                    raise CatBoostError('class "{}" is present in "class_names" but not in "class_weights" dictionary'.format(class_label))
                class_weights_list.append(class_weights_dict[class_label])
        else:
            class_labels_list = []
            for class_label, class_weight in class_weights_dict.items():
                class_labels_list.append(class_label)
                class_weights_list.append(class_weight)
            params['class_names'] = class_labels_list
        params['class_weights'] = class_weights_list
    _process_synonyms_groups(params)
    metric_period = None
    if 'metric_period' in params:
        metric_period = params['metric_period']
        del params['metric_period']
    verbose = None
    if 'verbose' in params:
        verbose = params['verbose']
        del params['verbose']
    logging_level = None
    if 'logging_level' in params:
        logging_level = params['logging_level']
        del params['logging_level']
    verbose_eval = None
    if 'verbose_eval' in params:
        verbose_eval = params['verbose_eval']
        del params['verbose_eval']
    silent = None
    if 'silent' in params:
        silent = params['silent']
        del params['silent']
    metric_period, verbose, logging_level = _process_verbose(metric_period, verbose, logging_level, verbose_eval, silent)
    if metric_period is not None:
        params['metric_period'] = metric_period
    if verbose is not None:
        params['verbose'] = verbose
    if logging_level is not None:
        params['logging_level'] = logging_level
    if 'used_ram_limit' in params:
        params['used_ram_limit'] = str(params['used_ram_limit'])