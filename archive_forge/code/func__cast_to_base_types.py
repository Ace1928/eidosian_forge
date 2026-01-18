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
def _cast_to_base_types(value):
    if isinstance(value, list):
        for index, element in enumerate(value):
            value[index] = _cast_to_base_types(element)
        return value
    if isinstance(value, ARRAY_TYPES[1:]):
        new_value = []
        for element in value:
            new_value.append(_cast_to_base_types(element))
        return new_value
    if isinstance(value, (Mapping, MutableMapping)):
        for key in list(value):
            value[key] = _cast_to_base_types(value[key])
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, INTEGER_TYPES):
        return int(value)
    if isinstance(value, FLOAT_TYPES):
        return float(value)
    if isinstance(value, tuple(set(PATH_TYPES) - set(STRING_TYPES))):
        return fspath(value)
    return value