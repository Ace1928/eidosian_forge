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
class SessionInterface(object):
    """Base class for implementations of TensorFlow client sessions."""

    @property
    def graph(self):
        """The underlying TensorFlow graph, to be used in building Operations."""
        raise NotImplementedError('graph')

    @property
    def sess_str(self):
        """The TensorFlow process to which this session will connect."""
        raise NotImplementedError('sess_str')

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """Runs operations in the session. See `BaseSession.run()` for details."""
        raise NotImplementedError('run')

    def partial_run_setup(self, fetches, feeds=None):
        """Sets up the feeds and fetches for partial runs in the session."""
        raise NotImplementedError('partial_run_setup')

    def partial_run(self, handle, fetches, feed_dict=None):
        """Continues the execution with additional feeds and fetches."""
        raise NotImplementedError('partial_run')