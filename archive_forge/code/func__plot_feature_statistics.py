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
def _plot_feature_statistics(statistics_by_feature, pool_names, feature_names, max_cat_features_on_plot):
    figs_with_names = []
    for feature_num in statistics_by_feature:
        feature_name = feature_names[feature_num]
        statistics = statistics_by_feature[feature_num]
        need_skip = 'cat_values' not in statistics[0].keys()
        if 'borders' in statistics[0].keys():
            for stats in statistics:
                if len(stats['borders']) > 0:
                    need_skip = False
        if need_skip:
            continue
        figs_with_names += _plot_feature_statistics_units(statistics, pool_names, feature_name, max_cat_features_on_plot)
    main_fig = figs_with_names[0][0]
    buttons = []
    for fig, feature_name in figs_with_names:
        buttons.append(dict(label=feature_name, method='update', args=[{'y': [data.y for data in fig.data]}, {'xaxis': fig.layout.xaxis}]))
    main_fig.update_layout(updatemenus=[dict(direction='down', pad={'r': 10, 't': 10}, showactive=True, x=0.25, xanchor='left', y=1.09, yanchor='top', buttons=buttons)], annotations=[dict(text='Statistics for feature', showarrow=False, x=0, xref='paper', y=1.05, yref='paper', align='left')])
    return main_fig