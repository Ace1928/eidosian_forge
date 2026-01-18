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
def _set_trained_model_attributes(self):
    setattr(self, '_is_fitted_', True)
    setattr(self, '_random_seed', self._object._get_random_seed())
    setattr(self, '_learning_rate', self._object._get_learning_rate())
    setattr(self, '_tree_count', self._object._get_tree_count())
    setattr(self, '_n_features_in', self._object._get_n_features_in())