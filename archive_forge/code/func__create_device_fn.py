import collections
import copy
import math
import re
from typing import Optional
from tensorflow.core.protobuf.tpu import optimization_parameters_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as elc
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.util.tf_export import tf_export
def _create_device_fn(hosts):
    """Create device_fn() to use with _create_partitioned_variables()."""

    def device_fn(op):
        """Returns the `device` for `op`."""
        part_match = re.match('.*/part_(\\d+)(/|$)', op.name)
        dummy_match = re.match('.*dummy_(\\d+).*', op.name)
        if not part_match and (not dummy_match):
            raise RuntimeError('Internal Error: Expected {} to contain /part_* or dummy_*'.format(op.name))
        if part_match:
            idx = int(part_match.group(1))
        else:
            idx = int(dummy_match.group(1))
        device = hosts[idx]
        logging.debug('assigning {} to {}.', op, device)
        return device
    return device_fn