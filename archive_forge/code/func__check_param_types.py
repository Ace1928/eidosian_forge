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
def _check_param_types(params):
    if not isinstance(params, (Mapping, MutableMapping)):
        raise CatBoostError('Invalid params type={}: must be dict().'.format(type(params)))
    if 'ctr_description' in params:
        if not isinstance(params['ctr_description'], Sequence):
            raise CatBoostError('Invalid ctr_description type={} : must be list of strings'.format(type(params['ctr_description'])))
    if 'ctr_target_border_count' in params:
        if not isinstance(params['ctr_target_border_count'], INTEGER_TYPES):
            raise CatBoostError('Invalid ctr_target_border_count type={} : must be integer type'.format(type(params['ctr_target_border_count'])))
    _cast_value_to_list_of_strings(params, 'custom_loss')
    _cast_value_to_list_of_strings(params, 'custom_metric')
    _cast_value_to_list_of_strings(params, 'per_float_feature_quantization')
    if 'monotone_constraints' in params:
        if not isinstance(params['monotone_constraints'], STRING_TYPES + ARRAY_TYPES + (dict,)):
            raise CatBoostError('Invalid `monotone_constraints` type={} : must be string or list of ints in range {{-1, 0, 1}} or dict.'.format(type(params['monotone_constraints'])))
    if 'feature_weights' in params:
        if not isinstance(params['feature_weights'], STRING_TYPES + ARRAY_TYPES + (dict,)):
            raise CatBoostError('Invalid `feature_weights` type={} : must be string or list of floats or dict.'.format(type(params['feature_weights'])))
    if 'first_feature_use_penalties' in params:
        if not isinstance(params['first_feature_use_penalties'], STRING_TYPES + ARRAY_TYPES + (dict,)):
            raise CatBoostError('Invalid `first_feature_use_penalties` type={} : must be string or list of floats or dict.'.format(type(params['first_feature_use_penalties'])))
    if 'per_object_feature_penalties' in params:
        if not isinstance(params['per_object_feature_penalties'], STRING_TYPES + ARRAY_TYPES + (dict,)):
            raise CatBoostError('Invalid `per_object_feature_penalties` type={} : must be string or list of floats or dict.'.format(type(params['per_object_feature_penalties'])))