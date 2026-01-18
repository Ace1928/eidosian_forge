import math
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch._decomp import register_decomposition
from torch._prims_common import DimsType, ShapeType, TensorLikeType
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper
def _fft_c2r(func_name: str, input: TensorLikeType, n: Optional[int], dim: int, norm: NormType, forward: bool) -> TensorLikeType:
    """Common code for performing any complex to real FFT (irfft or hfft)"""
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)
    last_dim_size = n if n is not None else 2 * (input.shape[dim] - 1)
    torch._check(last_dim_size >= 1, lambda: f'Invalid number of data points ({last_dim_size}) specified')
    if n is not None:
        input = _resize_fft_input(input, dims=dims, sizes=(last_dim_size // 2 + 1,))
    if forward:
        input = torch.conj(input)
    output = prims.fft_c2r(input, dim=dims, last_dim_size=last_dim_size)
    return _apply_norm(output, norm=norm, signal_numel=last_dim_size, forward=forward)