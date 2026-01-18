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
def _calc_leaf_indexes(self, data, ntree_start, ntree_end, thread_count, verbose):
    if ntree_end == 0:
        ntree_end = self.tree_count_
    data, _ = self._process_predict_input_data(data, 'calc_leaf_indexes', thread_count)
    return self._base_calc_leaf_indexes(data, ntree_start, ntree_end, thread_count, verbose)