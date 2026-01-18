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
def _check_transform_tags(self, tags, feature_names):
    if not isinstance(tags, dict):
        raise CatBoostError('Invalid feature_tags type={}: must be dict like.'.format(type(tags)))
    for tag_name, tag_features in tags.items():
        if not isinstance(tag_features, dict):
            raise CatBoostError('Invalid type of value in feature_tags by key {}, value type is {}: must be dict like.'.format(tag_name, type(tag_features)))
        if 'features' not in tag_features:
            raise CatBoostError("Invalid value in feature_tags by key {}, key 'features' is needed.".format(tag_name))
        if not isinstance(tag_features['features'], ARRAY_TYPES):
            raise CatBoostError('Invalid type of value in feature_tags by key {}, value type of features is {}: must be array like.'.format(tag_name, type(tag_features['features'])))
        if 'cost' not in tag_features:
            tag_features['cost'] = 1.0
        else:
            if not isinstance(tag_features['cost'], (INTEGER_TYPES, str)):
                raise CatBoostError('Invalid type of value in feature_tags by key {}, value type of cost is {}: must be integer.'.format(tag_name, type(tag_features['cost'])))
            tag_features['cost'] = int(tag_features['cost'])
        for idx in range(len(tag_features['features'])):
            if isinstance(tag_features['features'][idx], INTEGER_TYPES):
                pass
            elif isinstance(tag_features['features'][idx], str) and feature_names is not None:
                try:
                    feature_id = feature_names.index(tag_features['features'][idx])
                except ValueError:
                    raise CatBoostError('Unknown feature in tag {}: {}'.format(tag_name, tag_features['features'][idx]))
                tag_features['features'][idx] = feature_id
            else:
                raise CatBoostError('Invalid type of feature in tag {}, value type is {}: must be int or feature name.'.format(tag_name, type(tag_features['features'][idx])))
    return tags