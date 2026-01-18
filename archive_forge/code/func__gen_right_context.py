import math
from typing import List, Optional, Tuple
import torch
def _gen_right_context(self, input: torch.Tensor) -> torch.Tensor:
    T = input.shape[0]
    num_segs = math.ceil((T - self.right_context_length) / self.segment_length)
    right_context_blocks = []
    for seg_idx in range(num_segs - 1):
        start = (seg_idx + 1) * self.segment_length
        end = start + self.right_context_length
        right_context_blocks.append(input[start:end])
    right_context_blocks.append(input[T - self.right_context_length:])
    return torch.cat(right_context_blocks)