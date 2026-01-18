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
def _row_partitions_for_ragged_tensor(value, rank, dtype):
    """Returns the row partitions for a tf.RaggedTensor."""
    assert rank > 1
    value_row_partitions = value._nested_row_partitions[:rank - 1]
    if len(value_row_partitions) < rank - 1:
        value_row_partitions += _row_partitions_for_tensor(value.flat_values, rank - len(value_row_partitions), dtype)
    assert len(value_row_partitions) == rank - 1
    return value_row_partitions