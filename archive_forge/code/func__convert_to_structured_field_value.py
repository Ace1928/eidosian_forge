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
def _convert_to_structured_field_value(value):
    """Converts `value` to a Tensor, RaggedTensor, or StructuredTensor."""
    if isinstance(value, (tensor.Tensor, ragged_tensor.RaggedTensor, StructuredTensor)):
        return value
    elif ragged_tensor.is_ragged(value):
        return ragged_tensor.convert_to_tensor_or_ragged_tensor(value)
    elif isinstance(value, extension_type.ExtensionType):
        return value
    else:
        try:
            return ops.convert_to_tensor(value)
        except (ValueError, TypeError) as e:
            raise TypeError('Unexpected type for value in `fields`: %r' % value) from e