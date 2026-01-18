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
def _update_params_quantize_part(params, ignored_features, per_float_feature_quantization, border_count, feature_border_type, sparse_features_conflict_fraction, dev_efb_max_buckets, nan_mode, input_borders, task_type, used_ram_limit, random_seed, dev_max_subset_size_for_build_borders):
    if ignored_features is not None:
        params.update({'ignored_features': ignored_features})
    if per_float_feature_quantization is not None:
        params.update({'per_float_feature_quantization': per_float_feature_quantization})
    if border_count is not None:
        params.update({'border_count': border_count})
    if feature_border_type is not None:
        params.update({'feature_border_type': feature_border_type})
    if sparse_features_conflict_fraction is not None:
        params.update({'sparse_features_conflict_fraction': sparse_features_conflict_fraction})
    if dev_efb_max_buckets is not None:
        params.update({'dev_efb_max_buckets': dev_efb_max_buckets})
    if nan_mode is not None:
        params.update({'nan_mode': nan_mode})
    if input_borders is not None:
        params.update({'input_borders': input_borders})
    if task_type is not None:
        params.update({'task_type': task_type})
    if used_ram_limit is not None:
        params.update({'used_ram_limit': used_ram_limit})
    if random_seed is not None:
        params.update({'random_seed': random_seed})
    if dev_max_subset_size_for_build_borders is not None:
        params.update({'dev_max_subset_size_for_build_borders': dev_max_subset_size_for_build_borders})
    return params