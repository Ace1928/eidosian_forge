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
class BatchMetricCalcer(_MetricCalcerBase):

    def __init__(self, catboost, metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir):
        super(BatchMetricCalcer, self).__init__(catboost)
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
            delete_temp_dir_flag = True
        else:
            delete_temp_dir_flag = False
        if isinstance(metrics, STRING_TYPES) or isinstance(metrics, BuiltinMetric):
            metrics = [metrics]
        metrics = stringify_builtin_metrics_list(metrics)
        self._create_calcer(metrics, ntree_start, ntree_end, eval_period, thread_count, tmp_dir, delete_temp_dir_flag)