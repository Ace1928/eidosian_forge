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
@classmethod
def _from_pylist_of_dict(cls, pyval, keys, rank, typespec, path_so_far):
    """Converts python list `pyval` to a StructuredTensor with rank>1."""
    fields = dict(((key, []) for key in keys))
    for child in pyval:
        _pyval_update_fields(child, fields, 1)
    if typespec is None:
        shape = tensor_shape.TensorShape([None] * rank)
        for key, target in fields.items():
            fields[key] = cls._from_pyval(target, None, path_so_far + (key,))
    else:
        field_specs = typespec._fields
        if not isinstance(typespec, StructuredTensor.Spec) or set(fields) - set(field_specs):
            raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, pyval, typespec))
        shape = typespec._shape
        if shape.rank < rank:
            raise ValueError('Value at %r does not match typespec (rank mismatch): %r vs %r' % (path_so_far, pyval, typespec))
        for key, spec in field_specs.items():
            fields[key] = cls._from_pyval(fields.get(key, []), spec, path_so_far + (key,))
    try:
        if not fields and typespec is None:
            return StructuredTensor._from_pylist_of_empty_dict(pyval, rank)
        return StructuredTensor.from_fields(fields=fields, shape=shape, validate=False)
    except Exception as exc:
        raise ValueError('Error parsing path %r' % (path_so_far,)) from exc