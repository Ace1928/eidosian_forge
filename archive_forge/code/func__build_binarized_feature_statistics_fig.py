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
def _build_binarized_feature_statistics_fig(statistics_list, pool_names):
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        warnings.warn('To draw binarized feature statistics you should install plotly.')
        raise ImportError(str(e))
    pools_count = len(statistics_list)
    data = []
    statistics = statistics_list[0]
    if 'borders' in statistics.keys():
        if len(statistics['borders']) == 0:
            xaxis = go.layout.XAxis(title='Bins', tickvals=[0])
            return go.Figure(data=[], layout=_calc_feature_statistics_layout(go, xaxis, pools_count == 1))
        order = np.arange(len(statistics['objects_per_bin']))
        bar_width = 0.8
        xaxis = go.layout.XAxis(title='Bins', tickmode='array', tickvals=list(range(len(statistics['borders']) + 1)), ticktext=['(-inf, {:.4f}]'.format(statistics['borders'][0])] + ['({:.4f}, {:.4f}]'.format(val_1, val_2) for val_1, val_2 in zip(statistics['borders'][:-1], statistics['borders'][1:])] + ['({:.4f}, +inf)'.format(statistics['borders'][-1])], showticklabels=False)
    elif 'cat_values' in statistics.keys():
        order = np.argsort(statistics['objects_per_bin'])[::-1]
        bar_width = 0.2
        xaxis = go.layout.XAxis(title='Cat values', tickmode='array', tickvals=list(range(len(statistics['cat_values']))), ticktext=statistics['cat_values'][order], showticklabels=True)
    else:
        raise CatBoostError('Expected field "borders" or "cat_values" in binarized feature statistics')
    for i, statistics in enumerate(statistics_list):
        if pools_count == 1:
            name_suffix = ''
        else:
            name_suffix = ', {} pool'.format(pool_names[i])
        trace_1 = go.Scatter(y=statistics['mean_target'][order], mode='lines+markers', name='Mean target' + name_suffix, yaxis='y1', xaxis='x')
        trace_2 = go.Scatter(y=statistics['mean_prediction'][order], mode='lines+markers', line={'dash': 'dash'}, name='Mean prediction on each segment of feature values' + name_suffix, yaxis='y1', xaxis='x')
        if len(statistics['mean_weighted_target']) != 0:
            trace_3 = go.Scatter(y=statistics['mean_weighted_target'][order], mode='lines+markers', line={'dash': 'dot'}, name='Mean weighted target' + name_suffix, yaxis='y1', xaxis='x')
        if pools_count > 1:
            objects_in_pool = statistics['objects_per_bin'].sum()
            color_a = np.array([30, 150, 30])
            color_b = np.array([30, 30, 150])
            color = (color_a * i + color_b * (pools_count - 1 - i)) / float(pools_count - 1)
            color = color.astype(int)
            trace_4 = go.Bar(y=statistics['objects_per_bin'][order] / float(objects_in_pool), width=bar_width / pools_count, name='% pool objects in bin (total {})'.format(objects_in_pool) + name_suffix, yaxis='y2', xaxis='x', marker={'color': 'rgba({}, {}, {}, 0.4)'.format(*color)})
        else:
            trace_4 = go.Bar(y=statistics['objects_per_bin'][order], width=bar_width, name='Objects per bin' + name_suffix, yaxis='y2', xaxis='x', marker={'color': 'rgba(30, 150, 30, 0.4)'})
        trace_5 = go.Scatter(y=statistics['predictions_on_varying_feature'][order], mode='lines+markers', line={'dash': 'dashdot'}, name='Mean prediction with substituted feature' + name_suffix, yaxis='y1', xaxis='x')
        if len(statistics['mean_weighted_target']) != 0:
            data += [trace_1, trace_2, trace_3, trace_4, trace_5]
        else:
            data += [trace_1, trace_2, trace_4, trace_5]
    layout = _calc_feature_statistics_layout(go, xaxis, pools_count == 1)
    fig = go.Figure(data=data, layout=layout)
    return fig