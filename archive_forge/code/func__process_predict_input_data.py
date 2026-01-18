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
def _process_predict_input_data(self, data, parent_method_name, thread_count, label=None):
    if not self.is_fitted() or self.tree_count_ is None:
        raise CatBoostError('There is no trained model to use {}(). Use fit() to train model. Then use this method.'.format(parent_method_name))
    is_single_object = _is_data_single_object(data)
    if not isinstance(data, Pool):
        data = Pool(data=[data] if is_single_object else data, label=label, cat_features=self._get_cat_feature_indices() if not isinstance(data, FeaturesData) else None, text_features=self._get_text_feature_indices() if not isinstance(data, FeaturesData) else None, embedding_features=self._get_embedding_feature_indices() if not isinstance(data, FeaturesData) else None, thread_count=thread_count)
    return (data, is_single_object)