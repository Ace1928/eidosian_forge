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
class EFeaturesSelectionAlgorithm(Enum):
    """Use prediction values change as feature strength, eliminate batch of features at once"""
    RecursiveByPredictionValuesChange = 'RecursiveByPredictionValuesChange'
    'Use loss function change as feature strength, eliminate batch of features at each step'
    RecursiveByLossFunctionChange = 'RecursiveByLossFunctionChange'
    'Use shap values to estimate loss function change, eliminate features one by one'
    RecursiveByShapValues = 'RecursiveByShapValues'