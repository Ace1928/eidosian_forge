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
def _tensor_getitem(self, key):
    rank = self.rank
    if len(key) <= rank:
        new_fields = dict(((field_name, field_value.__getitem__(key)) for field_name, field_value in self._fields.items()))
        result_shape = self.shape.as_list()
        for d, k in enumerate(key):
            if isinstance(k, slice):
                if not (k.start is None and k.stop is None and (k.step is None)):
                    result_shape[d] = None
            elif isinstance(k, (int, tensor.Tensor)):
                result_shape[d] = -1
            elif k is None:
                raise ValueError('Slicing not supported for tf.newaxis')
            else:
                raise ValueError('Slicing not supported for %r' % k)
        result_shape = [d for d in result_shape if d != -1]
        return StructuredTensor.from_fields(new_fields, result_shape)
    else:
        if not isinstance(key[rank], compat.bytes_or_text_types):
            raise ValueError('Key for indexing a StructuredTensor must be a string')
        return self._fields[key[rank]].__getitem__(key[:rank] + key[rank + 1:])