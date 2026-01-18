import math
from typing import List, Optional, Tuple
import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains
def _unpack_state(self, state: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    past_length = state[3][0][0].item()
    past_left_context_length = min(self.left_context_length, past_length)
    past_mem_length = min(self.max_memory_size, math.ceil(past_length / self.segment_length))
    pre_mems = state[0][self.max_memory_size - past_mem_length:]
    lc_key = state[1][self.left_context_length - past_left_context_length:]
    lc_val = state[2][self.left_context_length - past_left_context_length:]
    conv_cache = state[4]
    return (pre_mems, lc_key, lc_val, conv_cache)