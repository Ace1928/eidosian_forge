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
class _TPUInferenceContext(control_flow_ops.XLAControlFlowContext):
    """A `ControlFlowContext` for nodes inside a TPU inference computation.

  The primary role of `_TPUInferenceContext` is to indicate the mode of
  operation and possibly sanity check operators inside a
  tpu.rewrite_for_inference() computation.
  """

    def __init__(self, name: Text, check_ops: bool=True):
        super(_TPUInferenceContext, self).__init__()
        self._name = name
        self._check_ops = check_ops

    def AddOp(self, op):
        self._AddOpInternal(op)

    def _AddOpInternal(self, op):
        if self._check_ops and op.type in _DENYLISTED_INFERENCE_OPS:
            raise NotImplementedError(f'Operation of type {op.type} ({op.name}) is not supported on the TPU for inference. Execution will fail if this op is used in the graph. Make sure your variables are using variable_scope.')
        if self._outer_context:
            self._outer_context.AddInnerOp(op)

    def AddValue(self, val):
        result = val
        if self._outer_context:
            result = self._outer_context.AddValue(val)
        return result

    def AddInnerOp(self, op):
        self._AddOpInternal(op)

    @property
    def grad_state(self):
        return None