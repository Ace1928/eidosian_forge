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
class _TrainCallbacksWrapper(object):

    def __init__(self, callbacks):
        self._callbacks = callbacks

    def after_iteration(self, info):
        for cb in self._callbacks:
            if not cb.after_iteration(info):
                return False
        return True