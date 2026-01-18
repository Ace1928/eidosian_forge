import math
from dataclasses import dataclass
from typing import (
import torch
def _materialize_causal_mask(shape: Tuple[int, ...], dtype: torch.dtype=torch.float32, device: Union[str, torch.device]='cpu', *, window_size: Optional[int]=None, from_bottomright: bool=False) -> torch.Tensor:
    create_as = dtype if dtype is not torch.bfloat16 else torch.float32
    tensor = torch.full(shape, dtype=create_as, fill_value=1, device=device)
    num_queries, num_keys = shape[-2:]
    shift = 0
    if from_bottomright:
        shift = num_keys - num_queries
    mask = torch.tril(tensor, diagonal=shift).to(dtype)
    if window_size is not None:
        mask = torch.triu(mask, diagonal=shift - window_size + 1)
    mask = torch.log(mask)
    return mask.to(dtype)