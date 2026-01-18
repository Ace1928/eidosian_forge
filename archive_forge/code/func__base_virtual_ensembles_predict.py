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
def _base_virtual_ensembles_predict(self, pool, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose):
    return self._object._base_virtual_ensembles_predict(pool, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose)