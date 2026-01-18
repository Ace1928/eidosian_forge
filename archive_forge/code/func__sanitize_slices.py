import collections
import functools
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def _sanitize_slices(slices, intended_shape, deficient_shape):
    """Restricts slices to avoid overflowing size-1 (broadcast) dimensions.

  Args:
    slices: iterable of slices received by `__getitem__`.
    intended_shape: int `Tensor` shape for which the slices were intended.
    deficient_shape: int `Tensor` shape to which the slices will be applied.
      Must have the same rank as `intended_shape`.
  Returns:
    sanitized_slices: Python `list` of slice objects.
  """
    sanitized_slices = []
    idx = 0
    for slc in slices:
        if slc is Ellipsis:
            if idx < 0:
                raise ValueError('Found multiple `...` in slices {}'.format(slices))
            num_remaining_non_newaxis_slices = sum((s is not array_ops.newaxis for s in slices[slices.index(Ellipsis) + 1:]))
            idx = -num_remaining_non_newaxis_slices
        elif slc is array_ops.newaxis:
            pass
        else:
            is_broadcast = intended_shape[idx] > deficient_shape[idx]
            if isinstance(slc, slice):
                start, stop, step = (slc.start, slc.stop, slc.step)
                if start is not None:
                    start = _prefer_static_where(is_broadcast, 0, start)
                if stop is not None:
                    stop = _prefer_static_where(is_broadcast, 1, stop)
                if step is not None:
                    step = _prefer_static_where(is_broadcast, 1, step)
                slc = slice(start, stop, step)
            else:
                slc = _prefer_static_where(is_broadcast, 0, slc)
            idx += 1
        sanitized_slices.append(slc)
    return sanitized_slices