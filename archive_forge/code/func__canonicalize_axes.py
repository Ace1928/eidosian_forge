import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export
def _canonicalize_axes(axes, rank):
    rank = _maybe_static(rank)
    if isinstance(rank, core.Tensor):
        canonicalizer = lambda axis: cond(axis < 0, lambda: axis + rank, lambda: axis)
    else:
        canonicalizer = lambda axis: axis + rank if axis < 0 else axis
    return [canonicalizer(axis) for axis in axes]