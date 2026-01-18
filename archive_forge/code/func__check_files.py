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
def _check_files(self, data, column_description, pairs):
    """
        Check files existence.
        """
    data = fspath(data)
    column_description = fspath(column_description)
    pairs = fspath(pairs)
    if data.find('://') == -1 and (not os.path.isfile(data)):
        raise CatBoostError("Invalid data path='{}': file does not exist.".format(data))
    if column_description is not None and column_description.find('://') == -1 and (not os.path.isfile(column_description)):
        raise CatBoostError("Invalid column_description path='{}': file does not exist.".format(column_description))
    if pairs is not None and pairs.find('://') == -1 and (not os.path.isfile(pairs)):
        raise CatBoostError("Invalid pairs path='{}': file does not exist.".format(pairs))