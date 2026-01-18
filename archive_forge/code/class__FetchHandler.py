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
class _FetchHandler(object):
    """Handler for structured fetches.

  Given a graph, a user-provided structure for fetches, and a feed dict, this
  class takes care of generating a list of tensor names to fetch and op names
  to run for a low level `run()` call.

  Given the results of the low level run call, this class can also rebuild a
  result structure matching the user-provided structure for fetches, but
  containing the corresponding results.
  """

    def __init__(self, graph, fetches, feeds, feed_handles=None):
        """Creates a fetch handler.

    Args:
      graph: Graph of the fetches.   Used to check for fetchability and to
        convert all fetches to tensors or ops as needed.
      fetches: An arbitrary fetch structure: singleton, list, tuple, namedtuple,
        or dict.
      feeds: A feed dict where keys are Tensors.
      feed_handles: A dict from feed Tensors to TensorHandle objects used as
        direct feeds.
    """
        with graph.as_default():
            self._fetch_mapper = _FetchMapper.for_fetch(fetches)
        self._fetches = []
        self._targets = []
        self._feeds = feeds
        self._feed_handles = feed_handles or {}
        self._ops = []
        self._fetch_handles = {}
        for fetch in self._fetch_mapper.unique_fetches():
            if isinstance(fetch, ops.Operation):
                self._assert_fetchable(graph, fetch)
                self._targets.append(fetch)
                self._ops.append(True)
            else:
                self._assert_fetchable(graph, fetch.op)
                self._fetches.append(fetch)
                self._ops.append(False)
            if isinstance(fetch, tensor.Tensor) and (fetch.op.type == 'GetSessionHandle' or fetch.op.type == 'GetSessionHandleV2'):
                self._fetch_handles[fetch.ref()] = fetch.op.inputs[0].dtype
        self._final_fetches = [x for x in self._fetches if x.ref() not in feeds]

    def _assert_fetchable(self, graph, op):
        if not graph.is_fetchable(op):
            raise errors.InaccessibleTensorError(f'Operation {op.name} has been marked as not fetchable. Typically this happens when it is defined in another function or code block. Use return values, explicit Python locals or TensorFlow collections to access it.')

    def fetches(self):
        """Return the unique names of tensors to fetch.

    Returns:
      A list of strings.
    """
        return self._final_fetches

    def targets(self):
        """Return the unique names of ops to run.

    Returns:
      A list of strings.
    """
        return self._targets

    def build_results(self, session, tensor_values):
        """Build results matching the original fetch shape.

    `tensor_values` must be a list of the same length as
    the one returned by `fetches()`, and holding the requested
    fetch values.

    This method builds a struct with the same shape as the original `fetches`
    passed to the constructor, in which the fetches are replaced by their
    fetched value.

    Args:
      session: The enclosing session.  Used for tensor handles.
      tensor_values: List of values matching the list returned by fetches().

    Returns:
      A structure of the same shape as the original `fetches` argument but
        containing tensors or None (for fetched ops).
    """
        full_values = []
        assert len(self._final_fetches) == len(tensor_values)
        i = 0
        j = 0
        for is_op in self._ops:
            if is_op:
                full_values.append(None)
            else:
                if self._fetches[i].ref() in self._feed_handles:
                    value = self._feed_handles[self._fetches[i].ref()].eval()
                else:
                    value = self._feeds.get(self._fetches[i].ref())
                if value is None:
                    value = tensor_values[j]
                    j += 1
                dtype = self._fetch_handles.get(self._fetches[i].ref())
                if dtype:
                    full_values.append(session_ops.TensorHandle(value, dtype, session))
                else:
                    full_values.append(value)
                i += 1
        assert j == len(tensor_values)
        return self._fetch_mapper.build_results(full_values)