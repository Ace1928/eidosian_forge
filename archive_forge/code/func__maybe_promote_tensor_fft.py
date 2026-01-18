import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _maybe_promote_tensor_fft(t: TensorLikeType, require_complex: bool=False) -> TensorLikeType:
    """Helper to promote a tensor to a dtype supported by the FFT primitives"""
    cur_type = t.dtype
    new_type = _promote_type_fft(cur_type, require_complex, t.device)
    return _maybe_convert_to_dtype(t, new_type)