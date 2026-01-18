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
def _to_subclass(model, subclass):
    """
    Convert a CatBoost model to a sklearn-compatible model.

    Parameters
    ----------
    model : CatBoost model
        a model to convert from

    subclass : an sklearn-compatible class
        a class to convert to : CatBoostClassifier, CatBoostRegressor or CatBoostRanker

    Returns
    -------
    a converted model : `subclass` type
        a model converted from the initial CatBoost `model` to a sklearn-compatible `subclass` model
    """
    if isinstance(model, subclass):
        return model
    if not isinstance(model, CatBoost):
        raise CatBoostError('model should be a subclass of CatBoost')
    converted_model = subclass.__new__(subclass)
    params = deepcopy(model._init_params)
    _process_synonyms(params)
    if 'loss_function' in params:
        subclass._check_is_compatible_loss(params['loss_function'])
    for attr in model.__dict__:
        setattr(converted_model, attr, getattr(model, attr))
    return converted_model