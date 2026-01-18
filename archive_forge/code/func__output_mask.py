import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _output_mask(op, input: Tensor, *args, **kwargs) -> Tensor:
    """Return output mask of masked operation applied to given arguments."""
    if callable(op):
        is_reduction = op.__name__ in {'sum', 'prod', 'amax', 'amin', 'argmax', 'argmin', 'mean', 'median', 'norm', 'var', 'std', 'logsumexp'}
        is_normalization = op.__name__ in {'softmax', 'log_softmax', 'softmin', 'normalize', 'cumsum', 'cumprod'}
        if is_reduction:
            if op.__name__ == 'norm':
                if args:
                    args = args[1:]
            dim = args[0] if args else kwargs.get('dim')
            outmask = _input_mask(input, *args, **kwargs)
            keepdim = kwargs.get('keepdim', False)
            dim_ = _canonical_dim(dim, input.ndim)
            return _any(outmask, dim_, bool(keepdim))
        elif is_normalization:
            return _input_mask(input, *args, **kwargs)
        else:
            raise ValueError(f'_output_mask expected masked operation (got callable {op.__module__}.{op.__name__})')
    else:
        raise ValueError(f'_output_mask expected masked operation (got {type(op).__name__} object)')