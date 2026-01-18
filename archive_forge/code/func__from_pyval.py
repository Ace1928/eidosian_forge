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
def _from_pyval(cls, pyval, typespec, path_so_far):
    """Helper function for from_pyval.


    Args:
      pyval: The nested Python structure that should be used to create the new
        `StructuredTensor`.
      typespec: A `StructuredTensor.Spec` specifying the expected type for each
        field. If not specified, then all nested dictionaries are turned into
        StructuredTensors, and all nested lists are turned into Tensors (if
        rank<2) or RaggedTensors (if rank>=2).
      path_so_far: the path of fields that led here (for error messages).

    Returns:
      A `StructuredTensor`.
    """
    if isinstance(pyval, dict):
        return cls._from_pydict(pyval, typespec, path_so_far)
    elif isinstance(pyval, (list, tuple)):
        keys = set()
        rank = _pyval_find_struct_keys_and_depth(pyval, keys)
        if rank is not None:
            return cls._from_pylist_of_dict(pyval, keys, rank, typespec, path_so_far)
        else:
            return cls._from_pylist_of_value(pyval, typespec, path_so_far)
    else:
        return cls._from_pyscalar(pyval, typespec, path_so_far)