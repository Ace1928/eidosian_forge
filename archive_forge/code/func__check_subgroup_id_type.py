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
def _check_subgroup_id_type(self, subgroup_id):
    """
        Check type of subgroup_id parameter.
        """
    if not isinstance(subgroup_id, ARRAY_TYPES):
        raise CatBoostError('Invalid subgroup_id type={}: must be array like.'.format(type(subgroup_id)))