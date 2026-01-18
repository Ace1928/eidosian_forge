import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _do_call(self, fn, *args):
    try:
        return fn(*args)
    except errors.OpError as e:
        message = compat.as_text(e.message)
        m = BaseSession._NODEDEF_NAME_RE.search(message)
        node_def = None
        op = None
        if m is not None:
            node_name = m.group(3)
            try:
                op = self._graph.get_operation_by_name(node_name)
                node_def = op.node_def
            except KeyError:
                pass
        message = error_interpolation.interpolate_graph(message, self._graph)
        if 'only supports NHWC tensor format' in message:
            message += '\nA possible workaround: Try disabling Grappler optimizer\nby modifying the config for creating the session eg.\nsession_config.graph_options.rewrite_options.disable_meta_optimizer = True'
        raise type(e)(node_def, op, message)