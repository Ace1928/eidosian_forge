import collections
from collections import OrderedDict
import contextlib
import functools
import gc
import itertools
import math
import os
import random
import re
import tempfile
import threading
import time
import unittest
from absl.testing import parameterized
import numpy as np
from google.protobuf import descriptor_pool
from google.protobuf import text_format
from tensorflow.core.config import flags
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_sanitizers
from tensorflow.python import tf2
from tensorflow.python.client import device_lib
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.client import session
from tensorflow.python.compat.compat import forward_compatibility_horizon
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import _test_metrics_util
from tensorflow.python.framework import config
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import gen_sync_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_ops  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import _pywrap_stacktrace_handler
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import _pywrap_util_port
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.tf_export import tf_export
class FakeEagerSession:
    """Fake session so tests that conditionally use placeholders can use eager.

  There are a number of tests that conditionally use placeholders for shape
  inference. The pattern is demonstrated here:

  ```python
  with self.cached_session() as sess:
    if static_shape:
      y = math_ops.matmul(x, ...)
      feed_dict = {}
    else:
      x_ph = array_ops.placeholder(...)
      y = math_ops.matmul(x_ph, ...)
      feed_dict = {x_ph: x}
    val = sess.run(y, feed_dict=feed_dict)
  ```

  Since the feed_dict is empty when not using placeholders we should be able to
  call self.evaluate(), however this requires rewriting the test case.
  This class should be considered a stop-gap solution to get tests running with
  eager with minimal changes to the actual test.
  """

    def __init__(self, test_case):
        self._test_case = test_case

    def run(self, fetches, *args, **kwargs):
        """Evaluate `fetches`.

    Fail if additional args are specified.

    Args:
      fetches: A Tensor or a nested list/tuple of Tensors.
      *args: Positional arguments
      **kwargs: Keyword arguments

    Raises:
      RuntimeError: If args or kwargs are specified.

    Returns:
      Tensors as numpy values.
    """
        feed_dict = kwargs.pop('feed_dict', {})
        if feed_dict:
            raise RuntimeError('feed_dict is not supported when eager execution is enabled (in this case, sess.run(t) is shorthand for t.numpy()')
        if args or kwargs:
            raise RuntimeError('Optional args are not supported when eager execution is enabled (in this case, sess.run(t) is shorthand for t.numpy()')
        return self._test_case.evaluate(fetches)