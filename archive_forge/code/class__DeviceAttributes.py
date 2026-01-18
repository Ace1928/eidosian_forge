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
class _DeviceAttributes(object):
    """Struct-like object describing a device's attributes.

  Each device has 3 key properties:
   - name: the fully-qualified TensorFlow path to the device. For
        example: /job:worker/replica:0/task:3/device:CPU:0
   - device_type: the type of the device (e.g. CPU, GPU, TPU, etc.)
   - memory_limit_bytes: the maximum amount of memory available on the device
        (in bytes).
  """

    def __init__(self, name, device_type, memory_limit_bytes, incarnation):
        self._name = device.canonical_name(name)
        self._device_type = device_type
        self._memory_limit_bytes = memory_limit_bytes
        self._incarnation = incarnation

    @property
    def name(self):
        return self._name

    @property
    def device_type(self):
        return self._device_type

    @property
    def memory_limit_bytes(self):
        return self._memory_limit_bytes

    @property
    def incarnation(self):
        return self._incarnation

    def __repr__(self):
        return '_DeviceAttributes(%s, %s, %d, %d)' % (self.name, self.device_type, self.memory_limit_bytes, self.incarnation)