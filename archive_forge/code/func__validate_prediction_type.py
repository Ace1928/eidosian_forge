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
def _validate_prediction_type(self, prediction_type, valid_prediction_types=('Class', 'RawFormulaVal', 'Probability', 'LogProbability', 'Exponent', 'RMSEWithUncertainty')):
    if not isinstance(prediction_type, STRING_TYPES):
        raise CatBoostError('Invalid prediction_type type={}: must be str().'.format(type(prediction_type)))
    if prediction_type not in valid_prediction_types:
        raise CatBoostError('Invalid value of prediction_type={}: must be {}.'.format(prediction_type, ', '.join(valid_prediction_types)))