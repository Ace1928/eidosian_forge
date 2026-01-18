import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
def _pack_state(self, next_k: torch.Tensor, next_v: torch.Tensor, update_length: int, mems: torch.Tensor, conv_cache: torch.Tensor, state: List[torch.Tensor]) -> List[torch.Tensor]:
    new_k = torch.cat([state[1], next_k])
    new_v = torch.cat([state[2], next_v])
    state[0] = torch.cat([state[0], mems])[-self.max_memory_size:]
    state[1] = new_k[new_k.shape[0] - self.left_context_length:]
    state[2] = new_v[new_v.shape[0] - self.left_context_length:]
    state[3] = state[3] + update_length
    state[4] = conv_cache
    return state