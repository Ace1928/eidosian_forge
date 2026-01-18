import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING
import numpy as np
import pyarrow as pa
from .. import config
from ..utils.py_utils import map_nested
from .formatting import TensorFormatter
def _recursive_tensorize(self, data_struct):
    import torch
    if hasattr(data_struct, '__array__') and (not isinstance(data_struct, torch.Tensor)):
        data_struct = data_struct.__array__()
    if isinstance(data_struct, np.ndarray):
        if data_struct.dtype == object:
            return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
    elif isinstance(data_struct, (list, tuple)):
        return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
    return self._tensorize(data_struct)