import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
def _split_right_context(self, utterance: torch.Tensor, right_context: torch.Tensor) -> torch.Tensor:
    T, B, D = right_context.size()
    if T % self.right_context_length != 0:
        raise ValueError('Tensor length should be divisible by its right context length')
    num_segments = T // self.right_context_length
    right_context_segments = right_context.reshape(num_segments, self.right_context_length, B, D)
    right_context_segments = right_context_segments.permute(0, 2, 1, 3).reshape(num_segments * B, self.right_context_length, D)
    pad_segments = []
    for seg_idx in range(num_segments):
        end_idx = min(self.state_size + (seg_idx + 1) * self.segment_length, utterance.size(0))
        start_idx = end_idx - self.state_size
        pad_segments.append(utterance[start_idx:end_idx, :, :])
    pad_segments = torch.cat(pad_segments, dim=1).permute(1, 0, 2)
    return torch.cat([pad_segments, right_context_segments], dim=1).permute(0, 2, 1)