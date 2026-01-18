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
def _process_synonyms_groups(params):
    _process_synonyms_group(['learning_rate', 'eta'], params)
    _process_synonyms_group(['border_count', 'max_bin'], params)
    _process_synonyms_group(['depth', 'max_depth'], params)
    _process_synonyms_group(['rsm', 'colsample_bylevel'], params)
    _process_synonyms_group(['random_seed', 'random_state'], params)
    _process_synonyms_group(['l2_leaf_reg', 'reg_lambda'], params)
    _process_synonyms_group(['iterations', 'n_estimators', 'num_boost_round', 'num_trees'], params)
    _process_synonyms_group(['od_wait', 'early_stopping_rounds'], params)
    _process_synonyms_group(['custom_metric', 'custom_loss'], params)
    _process_synonyms_group(['max_leaves', 'num_leaves'], params)
    _process_synonyms_group(['min_data_in_leaf', 'min_child_samples'], params)