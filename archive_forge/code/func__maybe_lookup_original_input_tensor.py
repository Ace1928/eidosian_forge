import collections
import threading
import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _maybe_lookup_original_input_tensor(graph, tensor):
    if graph and graph in _CHECK_NUMERICS_INPUT_LOOKUP and (tensor.name in _CHECK_NUMERICS_INPUT_LOOKUP[graph]):
        return _CHECK_NUMERICS_INPUT_LOOKUP[graph][tensor.name]
    else:
        return tensor