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
def _from_pylist_of_empty_dict(cls, pyval, rank):
    """Converts a pylist of empty dictionaries to StructuredTensors."""
    if rank == 0:
        return StructuredTensor.from_fields(fields={}, shape=(), validate=False)
    elif rank == 1:
        nrows = len(pyval)
        shape = (nrows,)
        return StructuredTensor.from_fields(fields={}, shape=shape, nrows=nrows)
    elif rank > 1:
        ragged_zeros = ragged_factory_ops.constant(_dicts_to_zeros(pyval))
        nrows = len(pyval)
        shape = tensor_shape.TensorShape([len(pyval)] + [None] * (rank - 1))
        return StructuredTensor.from_fields(fields={}, shape=shape, row_partitions=ragged_zeros._nested_row_partitions, nrows=nrows)