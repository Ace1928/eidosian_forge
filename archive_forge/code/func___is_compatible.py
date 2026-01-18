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
@staticmethod
def __is_compatible(a, b):
    """Returns true if the given type serializations compatible."""
    if isinstance(a, TypeSpec):
        return a.is_compatible_with(b)
    if not TypeSpec.__same_types(a, b):
        return False
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all((TypeSpec.__is_compatible(x, y) for x, y in zip(a, b)))
    if isinstance(a, dict):
        return len(a) == len(b) and sorted(a.keys()) == sorted(b.keys()) and all((TypeSpec.__is_compatible(a[k], b[k]) for k in a.keys()))
    if isinstance(a, (tensor_shape.TensorShape, dtypes.DType)):
        return a.is_compatible_with(b)
    return a == b