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
def _calc_feature_statistics_layout(go, xaxis, single_pool):
    return go.Layout(yaxis={'title': 'Prediction and target', 'side': 'left', 'overlaying': 'y2'}, yaxis2={'title': 'Objects per bin' if single_pool else '% pool objects in bin', 'side': 'right', 'position': 1.0}, xaxis=xaxis, legend={'bgcolor': 'rgba(0,0,0,0)', 'x': 1.07})