import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _promote_type_fft(dtype: torch.dtype, require_complex: bool, device: torch.device) -> torch.dtype:
    """Helper to promote a dtype to one supported by the FFT primitives"""
    if dtype.is_complex:
        return dtype
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    allowed_types = [torch.float32, torch.float64]
    maybe_support_half = device.type in ['cuda', 'meta'] and (not torch.version.hip)
    if maybe_support_half:
        allowed_types.append(torch.float16)
    torch._check(dtype in allowed_types, lambda: f'Unsupported dtype {dtype}')
    if require_complex:
        dtype = utils.corresponding_complex_dtype(dtype)
    return dtype