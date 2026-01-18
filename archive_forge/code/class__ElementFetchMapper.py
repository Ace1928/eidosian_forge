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
class _ElementFetchMapper(_FetchMapper):
    """Fetch mapper for singleton tensors and ops."""

    def __init__(self, fetches, contraction_fn):
        """Creates an _ElementFetchMapper.

    This is the fetch mapper used for leaves in the fetch struct.  Because of
    the expansions mechanism, a leaf can actually fetch more than one tensor.

    Also note that the fetches here can be just strings (tensor or op names) or
    any other object that the graph knows how to convert to a tensor, such as a
    Variable.  So we have to run each fetch through `as_graph_element()` to get
    the corresponding tensor or op.

    Args:
      fetches: List of objects, as returned by a fetch_fn defined in
        _REGISTERED_EXPANSIONS.
      contraction_fn: Callable as returned by a fetch_fn.
    """
        self._unique_fetches = []
        for fetch in fetches:
            try:
                self._unique_fetches.append(ops.get_default_graph().as_graph_element(fetch, allow_tensor=True, allow_operation=True))
            except TypeError as e:
                raise TypeError(f'Argument `fetch` = {fetch} has invalid type "{type(fetch).__name__}" must be a string or Tensor. ({str(e)})')
            except ValueError as e:
                raise ValueError(f'Argument `fetch` = {fetch} cannot be interpreted as a Tensor. ({str(e)})')
            except KeyError as e:
                raise ValueError(f'Argument `fetch` = {fetch} cannot be interpreted as a Tensor. ({str(e)})')
        self._contraction_fn = contraction_fn

    def unique_fetches(self):
        return self._unique_fetches

    def build_results(self, values):
        if not values:
            return None
        else:
            return self._contraction_fn(values)