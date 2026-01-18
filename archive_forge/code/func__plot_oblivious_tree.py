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
def _plot_oblivious_tree(self, splits, leaf_values):
    from graphviz import Digraph
    graph = Digraph()
    layer_size = 1
    current_size = 0
    for split_num in range(len(splits) - 1, -2, -1):
        for node_num in range(layer_size):
            if split_num >= 0:
                node_label = splits[split_num].replace('bin=', 'value>', 1).replace('border=', 'value>', 1)
                color = 'black'
                shape = 'ellipse'
            else:
                node_label = leaf_values[node_num]
                color = 'red'
                shape = 'rect'
            try:
                node_label = node_label.decode('utf-8')
            except Exception:
                pass
            graph.node(str(current_size), node_label, color=color, shape=shape)
            if current_size > 0:
                parent = (current_size - 1) // 2
                edge_label = 'Yes' if current_size % 2 == 0 else 'No'
                graph.edge(str(parent), str(current_size), edge_label)
            current_size += 1
        layer_size *= 2
    return graph