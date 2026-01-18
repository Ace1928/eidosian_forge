from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _enclosing_tpu_context_and_graph() -> Tuple[Any, Any]:
    """Returns the TPUReplicateContext and its associated graph."""
    graph = ops.get_default_graph()
    while graph is not None:
        context_ = graph._get_control_flow_context()
        while context_ is not None:
            if isinstance(context_, TPUReplicateContext):
                return (context_, graph)
            context_ = context_.outer_context
        graph = getattr(graph, 'outer_graph', None)
    raise ValueError("get_replicated_var_handle() called without TPUReplicateContext. This shouldn't happen. Please file a bug.")