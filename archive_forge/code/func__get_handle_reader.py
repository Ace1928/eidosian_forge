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
def _get_handle_reader(graph, handle, dtype):
    """Return a read subgraph for this handle."""
    graph_key = TensorHandle._get_reader_key(handle)
    result = graph._handle_readers.get(graph_key)
    if result is None:
        handle_device = TensorHandle._get_device_name(handle)
        with graph.as_default(), graph.device(handle_device):
            holder = array_ops.placeholder(dtypes.string)
            _register_handle_feeder(holder.graph, holder, dtype)
            reader = gen_data_flow_ops.get_session_tensor(holder, dtype)
        result = (holder, reader)
        graph._handle_readers[graph_key] = result
    return result