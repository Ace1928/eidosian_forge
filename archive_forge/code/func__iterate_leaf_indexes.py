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
def _iterate_leaf_indexes(self, data, ntree_start, ntree_end):
    if ntree_end == 0:
        ntree_end = self.tree_count_
    data, _ = self._process_predict_input_data(data, 'iterate_leaf_indexes', thread_count=-1)
    leaf_indexes_iterator = self._leaf_indexes_iterator(data, ntree_start, ntree_end)
    for leaf_index in leaf_indexes_iterator:
        yield leaf_index