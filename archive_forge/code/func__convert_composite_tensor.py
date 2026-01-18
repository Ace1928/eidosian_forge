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
def _convert_composite_tensor(value, expected_type, path, context):
    """Converts `value` to a value of type `expected_type`."""
    if context == _ConversionContext.SPEC:
        if not (isinstance(value, type_spec.TypeSpec) and _issubclass(value.value_type, expected_type)):
            raise TypeError(f'{''.join(path)}: expected a TypeSpec for {expected_type.__name__!r}, got {type(value).__name__!r}')
        return value
    if not isinstance(value, expected_type):
        raise TypeError(f'{''.join(path)}: expected {expected_type.__name__!r}, got {type(value).__name__!r}')
    return value