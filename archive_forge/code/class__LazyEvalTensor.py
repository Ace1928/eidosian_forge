import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _LazyEvalTensor(core.Tensor):
    """A Tensor-like object that only evaluates its thunk when used."""

    def __init__(self, thunk):
        """Initializes a _LazyEvalTensor object.

    Args:
      thunk: A callable. A thunk which computes the value of the tensor.
    """
        self._thunk = thunk
        self._master_tensor = thunk()

    def _as_tensor(self, dtype=None, name=None, as_ref=False):
        del name
        assert not as_ref
        assert dtype in [None, self.dtype]
        return self._thunk()