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
def _convert_to_catboost(models):
    """
    Convert _Catboost instances to Catboost ones
    """
    output_models = []
    for model in models:
        cb_model = CatBoost()
        cb_model._object = model
        cb_model._set_trained_model_attributes()
        output_models.append(cb_model)
    return output_models