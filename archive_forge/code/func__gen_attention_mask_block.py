import math
from typing import List, Optional, Tuple
import torch
def _gen_attention_mask_block(col_widths: List[int], col_mask: List[bool], num_rows: int, device: torch.device) -> torch.Tensor:
    if len(col_widths) != len(col_mask):
        raise ValueError('Length of col_widths must match that of col_mask')
    mask_block = [torch.ones(num_rows, col_width, device=device) if is_ones_col else torch.zeros(num_rows, col_width, device=device) for col_width, is_ones_col in zip(col_widths, col_mask)]
    return torch.cat(mask_block, dim=1)