import abc
import functools
from typing import Any, List, Optional, Sequence, Type
import warnings
import numpy as np
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal
from tensorflow.python.types import trace
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
def _with_tensor_ranks_only(self) -> 'TypeSpec':
    """Returns a TypeSpec compatible with `self`, with tensor shapes relaxed.

    Returns:
      A `TypeSpec` that is compatible with `self`, where any `TensorShape`
      information has been relaxed to include only tensor rank (and not
      the dimension sizes for individual axes).
    """

    def relax(value):
        if isinstance(value, TypeSpec):
            return value._with_tensor_ranks_only()
        elif isinstance(value, tensor_shape.TensorShape) and value.rank is not None:
            return tensor_shape.TensorShape([None] * value.rank)
        else:
            return value
    return self._deserialize(nest.map_structure(relax, self._serialize()))