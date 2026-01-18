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
def _convert_tuple(value, expected_type, path, context):
    """Converts `value` to a tuple with type `expected_type`."""
    if not isinstance(value, typing.Sequence):
        raise TypeError(f'{''.join(path)}: expected tuple, got {type(value).__name__!r}')
    element_types = type_annotations.get_generic_type_args(expected_type)
    if len(element_types) == 2 and element_types[1] is Ellipsis:
        return tuple([_convert_value(v, element_types[0], path + (f'[{i}]',), context) for i, v in enumerate(value)])
    else:
        if len(value) != len(element_types):
            raise TypeError(f'{''.join(path)}: expected tuple with length {len(element_types)}, got {type(value).__name__!r})')
        return tuple([_convert_value(v, t, path + (f'[{i}]',), context) for i, (v, t) in enumerate(zip(value, element_types))])