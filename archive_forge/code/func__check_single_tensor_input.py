import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def _check_single_tensor_input(tensor):
    """Check if the tensor is with a supported type."""
    if isinstance(tensor, np.ndarray):
        return
    if types.cupy_available():
        if isinstance(tensor, types.cp.ndarray):
            return
    if types.torch_available():
        if isinstance(tensor, types.th.Tensor):
            return
    raise RuntimeError("Unrecognized tensor type '{}'. Supported types are: np.ndarray, torch.Tensor, cupy.ndarray.".format(type(tensor)))