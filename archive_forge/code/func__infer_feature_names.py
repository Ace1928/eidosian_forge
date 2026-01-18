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
def _infer_feature_names(self, data_as_data_frame, embedding_features_data=None, embedding_features=None):
    non_embedding_data_feature_names = list(data_as_data_frame.columns)
    if embedding_features_data is not None:
        if isinstance(embedding_features_data, dict):
            embedding_feature_names = list(embedding_features_data.keys())
            if embedding_features is not None:
                if set(embedding_features) != set(embedding_feature_names):
                    raise CatBoostError('keys of embedding_features_data and embedding_features are different')
            return non_embedding_data_feature_names + embedding_feature_names
        else:
            if embedding_features is None:
                raise CatBoostError('embedding_features is not specified but embedding_features_data without feature names is present')
            if not all([isinstance(embedding_feature_id, INTEGER_TYPES) for embedding_feature_id in embedding_features]):
                raise CatBoostError('embedding_features contain feature names but embedding_features_data without feature names is present')
            embedding_features_set = set(embedding_features)
            feature_names = []
            non_embedding_feature_idx = 0
            for feature_idx in range(len(non_embedding_data_feature_names) + len(embedding_features)):
                if feature_idx in embedding_features_set:
                    feature_names.append('_embedding_feature_%i' % feature_idx)
                else:
                    feature_names.append(non_embedding_data_feature_names[non_embedding_feature_idx])
                    non_embedding_feature_idx += 1
            return feature_names
    else:
        return non_embedding_data_feature_names