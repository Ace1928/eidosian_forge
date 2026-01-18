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
@staticmethod
def _check_is_compatible_loss(loss_function):
    is_ranking = CatBoost._is_ranking_objective(loss_function)
    is_regression = CatBoost._is_regression_objective(loss_function)
    if is_regression:
        warnings.warn("Regression loss ('{}') ignores an important ranking parameter 'group_id'".format(loss_function), RuntimeWarning)
    if not (is_ranking or is_regression):
        raise CatBoostError("Invalid loss_function='{}': for ranker use YetiRank, YetiRankPairwise, StochasticFilter, StochasticRank, QueryCrossEntropy, QueryRMSE, QuerySoftMax, PairLogit, PairLogitPairwise. It's also possible to use a regression loss".format(loss_function))