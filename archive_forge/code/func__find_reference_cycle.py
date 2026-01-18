import collections
from collections import OrderedDict
import contextlib
import functools
import gc
import itertools
import math
import os
import random
import re
import tempfile
import threading
import time
import unittest
from absl.testing import parameterized
import numpy as np
from google.protobuf import descriptor_pool
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.client import session
from tensorflow.python.compat.compat import forward_compatibility_horizon
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _test_metrics_util
from tensorflow.python.framework import config
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import gen_sync_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_ops  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import _pywrap_stacktrace_handler
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import _pywrap_util_port
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.tf_export import tf_export
def _find_reference_cycle(objects, idx):

    def get_ignore_reason(obj, denylist):
        """Tests whether an object should be omitted from the dependency graph."""
        if len(denylist) > 100:
            return '<depth limit>'
        if tf_inspect.isframe(obj):
            if 'test_util.py' in tf_inspect.getframeinfo(obj)[0]:
                return '<test code>'
        for b in denylist:
            if b is obj:
                return '<test code>'
        if obj is denylist:
            return '<test code>'
        return None

    def describe(obj, denylist, leaves_only=False):
        """Returns a custom human-readable summary of obj.

    Args:
      obj: the value to describe.
      denylist: same as denylist in get_ignore_reason.
      leaves_only: boolean flag used when calling describe recursively. Useful
        for summarizing collections.
    """
        if get_ignore_reason(obj, denylist):
            return '{}{}'.format(get_ignore_reason(obj, denylist), type(obj))
        if tf_inspect.isframe(obj):
            return 'frame: {}'.format(tf_inspect.getframeinfo(obj))
        elif tf_inspect.ismodule(obj):
            return 'module: {}'.format(obj.__name__)
        elif leaves_only:
            return '{}, {}'.format(type(obj), id(obj))
        elif isinstance(obj, list):
            return 'list({}): {}'.format(id(obj), [describe(e, denylist, leaves_only=True) for e in obj])
        elif isinstance(obj, tuple):
            return 'tuple({}): {}'.format(id(obj), [describe(e, denylist, leaves_only=True) for e in obj])
        elif isinstance(obj, dict):
            return 'dict({}): {} keys'.format(id(obj), len(obj.keys()))
        elif tf_inspect.isfunction(obj):
            return 'function({}) {}; globals ID: {}'.format(id(obj), obj.__name__, id(obj.__globals__))
        else:
            return '{}, {}'.format(type(obj), id(obj))

    def build_ref_graph(obj, graph, reprs, denylist):
        """Builds a reference graph as <referrer> -> <list of referents>.

    Args:
      obj: The object to start from. The graph will be built by recursively
        adding its referrers.
      graph: Dict holding the graph to be built. To avoid creating extra
        references, the graph holds object IDs rather than actual objects.
      reprs: Auxiliary structure that maps object IDs to their human-readable
        description.
      denylist: List of objects to ignore.
    """
        referrers = gc.get_referrers(obj)
        denylist = denylist + (referrers,)
        obj_id = id(obj)
        for r in referrers:
            if get_ignore_reason(r, denylist) is None:
                r_id = id(r)
                if r_id not in graph:
                    graph[r_id] = []
                if obj_id not in graph[r_id]:
                    graph[r_id].append(obj_id)
                    build_ref_graph(r, graph, reprs, denylist)
                    reprs[r_id] = describe(r, denylist)

    def find_cycle(el, graph, reprs, path):
        """Finds and prints a single cycle in the dependency graph."""
        if el not in graph:
            return
        for r in graph[el]:
            if r in path:
                logging.error('Reference cycle sample:')
                for p in path + (r,):
                    logging.error(reprs.get(p, 'unknown object ' + str(p)))
                return True
            elif find_cycle(r, graph, reprs, path + (r,)):
                return True
        return False
    obj = objects[idx]
    graph = {}
    reprs = {}
    build_ref_graph(obj, graph, reprs, (objects, graph, reprs, get_ignore_reason, describe, build_ref_graph, find_cycle))
    for k in graph:
        if find_cycle(k, graph, reprs, ()):
            return True
    return False