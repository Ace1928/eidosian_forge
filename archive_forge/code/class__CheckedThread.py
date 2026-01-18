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
class _CheckedThread(object):
    """A wrapper class for Thread that asserts successful completion.

    This class should be created using the TensorFlowTestCase.checkedThread()
    method.
    """

    def __init__(self, testcase, target, args=None, kwargs=None):
        """Constructs a new instance of _CheckedThread.

      Args:
        testcase: The TensorFlowTestCase for which this thread is being created.
        target: A callable object representing the code to be executed in the
          thread.
        args: A tuple of positional arguments that will be passed to target.
        kwargs: A dictionary of keyword arguments that will be passed to target.
      """
        self._testcase = testcase
        self._target = target
        self._args = () if args is None else args
        self._kwargs = {} if kwargs is None else kwargs
        self._thread = threading.Thread(target=self._protected_run)
        self._exception = None
        self._is_thread_joined = False

    def _protected_run(self):
        """Target for the wrapper thread. Sets self._exception on failure."""
        try:
            self._target(*self._args, **self._kwargs)
        except Exception as e:
            self._exception = e

    def start(self):
        """Starts the thread's activity.

      This must be called at most once per _CheckedThread object. It arranges
      for the object's target to be invoked in a separate thread of control.
      """
        self._thread.start()

    def join(self):
        """Blocks until the thread terminates.

      Raises:
        self._testcase.failureException: If the thread terminates with due to
          an exception.
      """
        self._is_thread_joined = True
        self._thread.join()
        if self._exception is not None:
            self._testcase.fail('Error in checkedThread: %s' % str(self._exception))

    def is_alive(self):
        """Returns whether the thread is alive.

      This method returns True just before the run() method starts
      until just after the run() method terminates.

      Returns:
        True if the thread is alive, otherwise False.
      """
        return self._thread.is_alive()

    def check_termination(self):
        """Returns whether the checked thread was properly used and did terminate.

      Every checked thread should be "join"ed after starting, and before the
      test tears down. If it is not joined, it is possible the thread will hang
      and cause flaky failures in tests.

      Raises:
        self._testcase.failureException: If check_termination was called before
        thread was joined.

        RuntimeError: If the thread is not terminated. This means thread was not
        joined with the main thread.
      """
        if self._is_thread_joined:
            if self.is_alive():
                raise RuntimeError('Thread was not joined with main thread, and is still running when the test finished.')
        else:
            self._testcase.fail('A checked thread was not joined.')