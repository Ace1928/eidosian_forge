import numpy as np
from tensorflow.core.framework import resource_handle_pb2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _get_handle_mover(graph, feeder, handle):
    """Return a move subgraph for this pair of feeder and handle."""
    dtype = _get_handle_feeder(graph, feeder)
    if dtype is None:
        return None
    handle_device = TensorHandle._get_device_name(handle)
    if feeder.op.device == handle_device:
        return None
    graph_key = TensorHandle._get_mover_key(feeder, handle)
    result = graph._handle_movers.get(graph_key)
    if result is None:
        holder, reader = _get_handle_reader(graph, handle, dtype)
        with graph.as_default(), graph.device(feeder.op.device):
            mover = gen_data_flow_ops.get_session_handle(reader)
        result = (holder, mover)
        graph._handle_movers[graph_key] = result
    return result