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
def _get_default_prediction_type(self):
    params = deepcopy(self._init_params)
    _process_synonyms(params)
    loss_function = params.get('loss_function')
    if loss_function and isinstance(loss_function, str):
        if loss_function.startswith('Poisson') or loss_function.startswith('Tweedie'):
            return 'Exponent'
        if loss_function == 'RMSEWithUncertainty':
            return 'RMSEWithUncertainty'
    return 'RawFormulaVal'