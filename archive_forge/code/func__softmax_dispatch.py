import logging
from typing import Optional
import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd
from xformers.triton.k_softmax import _softmax, _softmax_backward
def _softmax_dispatch(x: torch.Tensor, log: bool, mask: Optional[torch.Tensor], causal: bool=False) -> torch.Tensor:
    global _triton_registered_warnings
    try:
        if torch.cuda.is_available() and x.is_cuda and (not _triton_registered_warnings):
            return _softmax_triton.apply(x, mask, log, causal)
    except RuntimeError as e:
        _triton_registered_warnings = True
        logger.warning('Triton softmax kernel register spillover or invalid image caught.Deactivating this kernel, please file an issue int the xFormers repository')
        logger.warning(e)
    if mask is not None:
        x = x + mask
    if causal:
        x = x + torch.triu(torch.full_like(x, float('-inf')), diagonal=1)
    if log:
        return torch.log_softmax(x, dim=-1)
    else:
        return torch.softmax(x, dim=-1)