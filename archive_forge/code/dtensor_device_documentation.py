import contextlib
import logging
import threading
from typing import Any, List, Sequence, Set
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import _pywrap_utils
Sets a default output layout for all ops in the scope.

    Note: This is an internal helper method, which is not user facing api.

    Useful for requesting a specific layout for ops which would have no inferred
    layout, e.g. tf.zeros.

    Caveats:

    - Currently only affects the first output of an op. For Op with multiple
      outputs, this does not support yet.

    - All Ops in the scope will be attached with the same layout. This might not
      be valid as the rank is different. The current suggestion is: Try to wrap
      the raw op wheneven possible.

    Args:
      layout: A Layout for the outputs of all operations in this scope.

    Yields:
      Nothing.
    