import collections
import collections.abc
import enum
import typing
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.util import type_annotations
def _convert_union(value, expected_type, path, context):
    """Converts `value` to a value with any of the types in `expected_type`."""
    for type_option in type_annotations.get_generic_type_args(expected_type):
        try:
            return _convert_value(value, type_option, path, context)
        except TypeError:
            pass
    raise TypeError(f'{''.join(path)}: expected {expected_type!r}, got {type(value).__name__!r}')