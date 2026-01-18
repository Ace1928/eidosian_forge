import functools
from typing import Any, Callable, Dict, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save_context
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _raise_error_for_inputs_not_on_cpu(self, flat_inputs, flat_paths):
    """Checks all tensors in features to see are placed on the CPU."""

    def check_device(path, device_string):
        spec = tf_device.DeviceSpec.from_string(device_string)
        if spec.device_type == 'TPU':
            raise ValueError("Received input tensor {} which is on a TPU input device {}. Input tensors for TPU embeddings must be placed on the CPU. Please ensure that your dataset is prefetching tensors to the host by setting the 'experimental_fetch_to_device' option of the dataset distribution function. See the documentation of the enqueue method for an example.".format(path, device_string))
    for input_tensor, input_path in zip(flat_inputs, flat_paths):
        if nest.is_nested_or_composite(input_tensor):
            input_tensors = nest.flatten(input_tensor, expand_composites=True)
        else:
            input_tensors = [input_tensor]
        for t in input_tensors:
            if t.op.type == 'Identity' and t.op.inputs[0].op.type == 'TPUReplicatedInput':
                for tensor in t.op.inputs[0].op.inputs:
                    check_device(input_path, tensor.device)
            else:
                check_device(input_path, t.device)