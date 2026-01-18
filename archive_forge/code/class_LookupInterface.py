import collections
import functools
import uuid
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_lookup_ops import *
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import compat as compat_util
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
class LookupInterface(resource.TrackableResource):
    """Represent a lookup table that persists across different steps."""

    def __init__(self, key_dtype, value_dtype):
        """Construct a lookup table interface.

    Args:
      key_dtype: The table key type.
      value_dtype: The table value type.
    """
        self._key_dtype = dtypes.as_dtype(key_dtype)
        self._value_dtype = dtypes.as_dtype(value_dtype)
        super(LookupInterface, self).__init__()

    def _create_resource(self):
        raise NotImplementedError

    @property
    def key_dtype(self):
        """The table key dtype."""
        return self._key_dtype

    @property
    def value_dtype(self):
        """The table value dtype."""
        return self._value_dtype

    @property
    def name(self):
        """The name of the table."""
        return NotImplementedError

    def size(self, name=None):
        """Compute the number of elements in this table."""
        raise NotImplementedError

    def lookup(self, keys, name=None):
        """Looks up `keys` in a table, outputs the corresponding values."""
        raise NotImplementedError

    def __getitem__(self, keys):
        """Looks up `keys` in a table, outputs the corresponding values."""
        return self.lookup(keys)