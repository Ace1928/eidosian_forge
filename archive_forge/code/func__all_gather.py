import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
def _all_gather(self, input_tensor: core.TensorLike, options: Optional[collective_util.Options]) -> core.Tensor:
    """All-gather a dense tensor.

    Args:
      input_tensor: a dense tensor. It must have the same shape on all replicas.
      options: an optional tf.distribute.experimental.CommunicationOptions. If
        provided, it overrides the default options.

    Returns:
      The reduced tensor.
    """
    instance_key = self._next_instance_key()
    options = self._options.merge(options)
    ordering_token = self._get_ordering_token()
    with ops.device(self._device):
        return collective_ops.all_gather_v2(input_tensor, self._group_size, self._group_key, instance_key, communication_hint=options.implementation.value, timeout=options.timeout_seconds, ordering_token=ordering_token)