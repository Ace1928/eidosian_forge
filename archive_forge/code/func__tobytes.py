import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from safetensors import deserialize, safe_open, serialize, serialize_file
def _tobytes(tensor: torch.Tensor, name: str) -> bytes:
    if tensor.layout != torch.strided:
        raise ValueError(f'You are trying to save a sparse tensor: `{name}` which this library does not support. You can make it a dense tensor before saving with `.to_dense()` but be aware this might make a much larger file than needed.')
    if not tensor.is_contiguous():
        raise ValueError(f"You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means you are trying to save tensors which are reference of each other in which case it's recommended to save only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to pack it before saving.")
    if tensor.device.type != 'cpu':
        tensor = tensor.to('cpu')
    import ctypes
    import numpy as np
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = _SIZE[tensor.dtype]
    total_bytes = length * bytes_per_item
    ptr = tensor.data_ptr()
    if ptr == 0:
        return b''
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(newptr, (total_bytes,))
    if sys.byteorder == 'big':
        NPDTYPES = {torch.int64: np.int64, torch.float32: np.float32, torch.int32: np.int32, torch.bfloat16: np.float16, torch.float16: np.float16, torch.int16: np.int16, torch.uint8: np.uint8, torch.int8: np.int8, torch.bool: bool, torch.float64: np.float64, _float8_e4m3fn: np.uint8, _float8_e5m2: np.uint8}
        npdtype = NPDTYPES[tensor.dtype]
        data = data.view(npdtype).byteswap(inplace=False)
    return data.tobytes()