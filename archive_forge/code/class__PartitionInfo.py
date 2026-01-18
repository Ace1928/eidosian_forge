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
class _PartitionInfo:
    """Holds partition info used by initializer functions."""
    __slots__ = ['_full_shape', '_var_offset']

    def __init__(self, full_shape, var_offset):
        """Constructor.

    Args:
      full_shape: Tuple or list of `int` indicating the full combined shape of
        the partitioned variables.
      var_offset: Tuple or list of `int` specifying offset of this partition
        with respect to the full variable for each dimension.

    Raises:
      TypeError: If `full_shape` or `var_offset` is not a sequence.
      ValueError: If `full_shape` or `var_offset` differ in length. If
        `var_offset` exceeds `full_shape` in any dimension.
    """
        if not isinstance(full_shape, (list, tuple)):
            raise TypeError('`full_shape` must be a sequence (like tuple or list) instead of ' + type(full_shape).__name__)
        if not isinstance(var_offset, (list, tuple)):
            raise TypeError('`var_offset` must be a sequence (like tuple or list) instead of ' + type(var_offset).__name__)
        if len(var_offset) != len(full_shape):
            raise ValueError('Expected equal length, but `var_offset` is of length {} while full_shape is of length {}.'.format(len(var_offset), len(full_shape)))
        for offset, shape in zip(var_offset, full_shape):
            if offset < 0 or offset >= shape:
                raise ValueError('Expected 0 <= offset < shape but found offset={}, shape={} for var_offset={}, full_shape={}'.format(offset, shape, var_offset, full_shape))
        self._full_shape = full_shape
        self._var_offset = var_offset

    @property
    def full_shape(self):
        return self._full_shape

    @property
    def var_offset(self):
        return self._var_offset

    def single_offset(self, shape):
        """Returns the offset when the variable is partitioned in at most one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the offset in the dimension along which the variable is
       partitioned. Returns 0 if the variable is not being partitioned.

    Raises:
      ValueError: Depending on self.single_slice_dim().
    """
        single_slice_dim = self.single_slice_dim(shape)
        if single_slice_dim is None:
            return 0
        return self.var_offset[single_slice_dim]

    def single_slice_dim(self, shape):
        """Returns the slice dim when the variable is partitioned only in one dim.

    Args:
      shape: Tuple or list of `int` indicating the shape of one specific
        variable partition.

    Returns:
      `int` representing the dimension that the variable is partitioned in, or
      `None` if the variable doesn't seem to be partitioned at all.

    Raises:
      TypeError: If `shape` is not a sequence.
      ValueError: If `shape` is not the same length as `self.full_shape`. If
        the variable is partitioned in more than one dimension.
    """
        if not isinstance(shape, (tuple, list)):
            raise TypeError('`shape` must be a sequence (like tuple or list) instead of ' + type(shape).__name__)
        if len(shape) != len(self.full_shape):
            raise ValueError('Expected equal length, but received shape={} of length {} while self.full_shape={} is of length {}.'.format(shape, len(shape), self.full_shape, len(self.full_shape)))
        for i in range(len(shape)):
            if self.var_offset[i] + shape[i] > self.full_shape[i]:
                raise ValueError('With self.var_offset={}, a partition of shape={} would exceed self.full_shape={} in dimension {}.'.format(self.var_offset, shape, self.full_shape, i))
        slice_dim = None
        for i in range(len(shape)):
            if shape[i] == self.full_shape[i]:
                continue
            if slice_dim is not None:
                raise ValueError('Cannot use single_slice_dim() with shape={} and self.full_shape={} since slice dim could be either dimension {} or {}.'.format(shape, self.full_shape, i, slice_dim))
            slice_dim = i
        return slice_dim