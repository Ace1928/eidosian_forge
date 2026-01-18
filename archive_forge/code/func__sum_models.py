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
def _sum_models(self, models_base, weights=None, ctr_merge_policy='IntersectingCountersAverage'):
    if weights is None:
        weights = [1.0 for _ in models_base]
    models_inner = [model._object for model in models_base]
    self._object._sum_models(models_inner, weights, ctr_merge_policy)
    setattr(self, '_random_seed', 0)
    setattr(self, '_learning_rate', 0)
    setattr(self, '_tree_count', self._object._get_tree_count())