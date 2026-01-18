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
def _check_pairs_value(self, pairs):
    """
        Check values in pairs parameter. Must be int indices.
        """
    for pair_id, pair in enumerate(pairs):
        if len(pair) != 2:
            raise CatBoostError("Length of pairs[{}] isn't equal to 2.".format(pair_id))
        for i, index in enumerate(pair):
            if not isinstance(index, INTEGER_TYPES):
                raise CatBoostError('Invalid pairs[{}][{}] = {} value type={}: must be int().'.format(pair_id, i, index, type(index)))