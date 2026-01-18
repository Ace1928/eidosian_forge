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
class Spec:
    """A spec for StructuredTensor."""

    def __validate__(self):
        assert self._ragged_shape is not None

    @classmethod
    def _from_fields_and_rank(cls, fields, rank):
        """Creates a spec of a StructuredTensor with fields and rank."""
        shape = None
        for k, v in fields.items():
            field_shape_untruncated = _dynamic_ragged_shape_spec_from_spec(v)
            if field_shape_untruncated is None:
                raise ValueError(f'Cannot convert spec of {k}.')
            untruncated_rank = field_shape_untruncated.rank
            if untruncated_rank is not None and untruncated_rank < rank:
                raise ValueError(f'Rank of field {k} is {untruncated_rank}, but must be at least {rank}.')
            field_shape = field_shape_untruncated._truncate(rank)
            if shape is None:
                shape = field_shape
            else:
                shape = shape._merge_with(field_shape)
        return StructuredTensor.Spec(_ragged_shape=shape, _fields=fields)

    @classmethod
    def _from_shape(cls, shape: dynamic_ragged_shape.DynamicRaggedShape) -> 'StructuredTensor.Spec':
        """Creates the spec of an empty StructuredTensor."""
        return StructuredTensor.Spec(_ragged_shape=shape, _fields={})

    @property
    def _shape(self) -> tensor_shape.TensorShape:
        return self._ragged_shape._to_tensor_shape()

    @property
    def _field_specs(self) -> Dict[str, type_spec.TypeSpec]:
        return self._fields

    @property
    def shape(self) -> tensor_shape.TensorShape:
        return self._shape

    @property
    def rank(self):
        return self._ragged_shape.rank