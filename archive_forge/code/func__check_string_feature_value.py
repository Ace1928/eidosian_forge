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
def _check_string_feature_value(self, features, features_count, features_name):
    """
        Check values in cat_feature parameter. Must be int indices.
        """
    for indx, feature in enumerate(features):
        if not isinstance(feature, INTEGER_TYPES):
            raise CatBoostError('Invalid {}[{}] = {} value type={}: must be int().'.format(features_name, indx, feature, type(feature)))
        if feature >= features_count:
            raise CatBoostError('Invalid {}[{}] = {} value: index must be < {}.'.format(features_name, indx, feature, features_count))