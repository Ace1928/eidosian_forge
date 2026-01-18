import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def _make_sparse_tensor(name: str) -> SparseTensorProto:
    dense_shape = [3, 3]
    linear_indices = [2, 3, 5]
    sparse_values = [1.7, 0.4, 0.9]
    values_tensor = helper.make_tensor(name=name + '_values', data_type=TensorProto.FLOAT, dims=[len(sparse_values)], vals=np.array(sparse_values).astype(np.float32), raw=False)
    indices_tensor = helper.make_tensor(name=name + '_idx', data_type=TensorProto.INT64, dims=[len(linear_indices)], vals=np.array(linear_indices).astype(np.int64), raw=False)
    return helper.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)