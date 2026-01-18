import abc
import collections
import contextlib
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import object_identity
def _lookup(self, x=None, y=None, kwargs=None):
    """Helper which retrieves mapping info from forward/inverse dicts."""
    mapping = _Mapping(x=x, y=y, kwargs=kwargs)
    if mapping.x is not None:
        return self._from_x.get(mapping.x_key, mapping)
    if mapping.y is not None:
        return self._from_y.get(mapping.y_key, mapping)
    return mapping