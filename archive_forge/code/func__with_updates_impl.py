import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _with_updates_impl(self, error_prefix: Tuple[str, ...], updates: List[Tuple[FieldName, Union[_FieldValue, _FieldFn]]], validate: bool) -> 'StructuredTensor':
    """Recursive part of `with_updates` implementation."""
    new_fields = dict(self._fields)

    def name_fullpath(name: Sequence[str]) -> str:
        return str(error_prefix + (name,))

    def apply_value(name: str, value: Union[_FieldValue, _FieldFn]) -> _FieldValue:
        if callable(value):
            if name not in new_fields:
                raise ValueError('`StructuredTensor.with_updates` cannot update the field {} because a transforming function was given, but that field does not already exist.'.format(name_fullpath(name)))
            value = value(new_fields[name])
        return value
    for name, value in updates:
        if not name or not name[0]:
            raise ValueError('`StructuredTensor.with_updates` does not allow empty names {}.'.format(name_fullpath(name)))
        if len(name) == 1:
            name = name[0]
            if value is None:
                if name not in new_fields:
                    raise ValueError('`StructuredTensor.with_updates` cannot delete field {} because it is not present.'.format(name_fullpath(name)))
                new_fields.pop(name)
            else:
                new_fields[name] = apply_value(name, value)
        else:
            prefix = name[0]
            suffix = name[1:]
            if prefix not in new_fields:
                raise ValueError('`StructuredTensor.with_updates` cannot create new sub-field {} if parent field {} is not set.'.format(error_prefix + tuple(name), name_fullpath(prefix)))
            current_value = new_fields[prefix]
            if not isinstance(current_value, StructuredTensor):
                raise ValueError('`StructuredTensor.with_updates` cannot create new sub-field {} if parent structure {} is not a `StructuredTensor` that can contain sub-structures -- it is a `{}`.'.format(error_prefix + tuple(name), name_fullpath(prefix), type(current_value)))
            one_update = [(suffix, value)]
            value = current_value._with_updates_impl(error_prefix + (prefix,), one_update, validate)
            new_fields[prefix] = value
    try:
        return StructuredTensor.from_fields(new_fields, shape=self.shape, row_partitions=self.row_partitions, nrows=self.nrows(), validate=validate)
    except ValueError as e:
        msg = '`StructuredTensor.with_updates` failed'
        if error_prefix:
            msg = '{} for field {}'.format(msg, error_prefix)
        raise ValueError(msg) from e