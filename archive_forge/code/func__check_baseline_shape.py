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
def _check_baseline_shape(self, baseline, samples_count):
    """
        Check baseline length and dimension.
        """
    if len(baseline) != samples_count:
        raise CatBoostError('Length of baseline={} and length of data={} are different.'.format(len(baseline), samples_count))
    if not isinstance(baseline[0], Iterable) or isinstance(baseline[0], STRING_TYPES):
        raise CatBoostError('Baseline must be 2 dimensional data, 1 column for each class.')
    try:
        if np.array(baseline).dtype not in (np.dtype('float'), np.dtype('float32'), np.dtype('int')):
            raise CatBoostError()
    except CatBoostError:
        raise CatBoostError('Invalid baseline value type={}: must be float or int.'.format(np.array(baseline).dtype))