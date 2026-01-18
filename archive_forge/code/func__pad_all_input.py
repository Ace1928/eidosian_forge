import collections
import enum
from typing import Any, Callable, Iterable, List, Optional, Text, Tuple, Union
from absl import logging
import numpy as np
from tensorflow.compiler.tf2xla.python import xla as tf2xla
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf.tpu import dynamic_padding_pb2 as dynamic_padding
from tensorflow.core.protobuf.tpu import tpu_embedding_configuration_pb2 as embedding_pb2
from tensorflow.python import tf2
from tensorflow.python.compiler.xla import xla
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import tpu_name_util
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _pad_all_input(inputs: Iterable[core_types.Tensor], padded_shapes: List[Optional[tensor_shape.TensorShape]], padding_spec: PaddingSpec) -> Tuple[List[List[Any]], List[dynamic_padding.PaddingMap]]:
    """Pad all input tensors given padded_shapes.

  The real shape tensors will be concatenated with the padded original inputs.

  Args:
    inputs: The original inputs.
    padded_shapes: A list of padded shapes for each input. If an entry is None,
      no padding is performed.
    padding_spec: An enum specified by `tpu.PaddingSpec`. This describes the
      padding policy when the `inputs` to `tf.tpu.replicate` is dynamic.
      One usage is to enable automatic bucketizing on the inputs by setting the
      value to `tpu.PaddingSpec.POWER_OF_TWO`, which can help to reduce the
      recompilation in the XLA side.

  Returns:
    The padded inputs and a PaddingMap list which maps the padded input
    dimension to the real shape argument index.
  """
    maximum_static_shapes = []
    need_padding = []
    input_shape_tensors = []
    for core_idx, inputs_per_core in enumerate(inputs):
        for idx, input_tensor in enumerate(inputs_per_core):
            input_shape = input_tensor.get_shape().as_list()
            if core_idx == 0:
                input_shape_tensors.append([])
                maximum_static_shapes.append(input_shape)
                need_padding.append(np.full_like(input_shape, False, dtype=bool))
            else:
                for i, s in enumerate(input_shape):
                    if s is None or s != maximum_static_shapes[idx][i]:
                        need_padding[idx][i] = True
                maximum_static_shapes[idx] = max(input_shape, maximum_static_shapes[idx])
            real_input_shape = array_ops.shape(input_tensor)
            real_input_shape.op._set_attr(_POST_DEVICE_REWRITE_ATTR, attr_value_pb2.AttrValue(b=True))
            input_shape_tensors[idx].append(real_input_shape)
    maximum_shapes = []
    for shapes_per_input in input_shape_tensors:
        maximum_shapes.append(math_ops.reduce_max(array_ops_stack.stack(shapes_per_input), axis=0))
    padded_inputs = []
    real_shapes = []
    padding_maps = []
    for core_idx, inputs_per_core in enumerate(inputs):
        padded_inputs.append([])
        real_shapes.append([])
        real_shape_idx = len(inputs_per_core) - 1
        for idx, input_tensor in enumerate(inputs_per_core):
            input_shape_tensor = input_shape_tensors[idx][core_idx]
            input_shape = input_tensor.get_shape().as_list()
            padded_shape = padded_shapes[idx]
            if any(need_padding[idx]) and padded_shape is not None:
                for i, s in enumerate(input_shape):
                    if need_padding[idx][i]:
                        if core_idx == 0:
                            real_shape_idx += 1
                            padding_map = dynamic_padding.PaddingMap()
                            padding_map.arg_index = idx
                            padding_map.shape_index = i
                            padding_map.padding_arg_index = real_shape_idx
                            padding_maps.append(padding_map)
                        real_shapes[core_idx].append(math_ops.cast(input_shape_tensor[i], dtypes.int32))
                paddings = []
                for i, s in enumerate(padded_shape.dims):
                    if need_padding[idx][i]:
                        minimum_dynamic_dim_size = 2
                        if s.value is not None:
                            max_dim_size = max(s.value, minimum_dynamic_dim_size)
                        else:
                            max_dim_size = math_ops.maximum(maximum_shapes[idx][i], minimum_dynamic_dim_size)
                            if padding_spec == PaddingSpec.POWER_OF_TWO:
                                max_dim_size = _ceil_to_pow_of_n(max_dim_size, 2)
                        padding = [0, max_dim_size - input_shape_tensor[i]]
                    else:
                        padding = [0, 0]
                    paddings.append(padding)
                if input_tensor.get_shape().is_fully_defined():
                    padded_input = cond.cond(array_ops.constant(True), lambda: array_ops.pad(input_tensor, paddings), lambda: input_tensor)
                else:
                    padded_input = array_ops.pad(input_tensor, paddings)
                padded_input.op._set_attr(_POST_DEVICE_REWRITE_ATTR, attr_value_pb2.AttrValue(b=True))
                padded_inputs[core_idx].append(padded_input)
            else:
                padded_inputs[core_idx].append(input_tensor)
    num_replicas = len(padded_inputs)
    for i in range(num_replicas):
        padded_inputs[i].extend(real_shapes[i])
    return (padded_inputs, padding_maps)