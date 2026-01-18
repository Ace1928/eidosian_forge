import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
def _merge_right_context(self, right_context: torch.Tensor, B: int) -> torch.Tensor:
    right_context = right_context.reshape(-1, B, self.input_dim, self.right_context_length)
    right_context = right_context.permute(0, 3, 1, 2)
    return right_context.reshape(-1, B, self.input_dim)