import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from vllm._C import ops
def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
    pos_freqs = self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
    low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.rotary_dim, self.base, self.max_position_embeddings)
    inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)) * self.extrapolation_factor
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    return inv_freq