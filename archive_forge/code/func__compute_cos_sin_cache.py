import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from vllm._C import ops
def _compute_cos_sin_cache(self) -> torch.Tensor:
    inv_freq = self._compute_inv_freq(self.scaling_factor)
    t = torch.arange(self.max_position_embeddings * self.scaling_factor, dtype=torch.float32)
    freqs = torch.einsum('i,j -> ij', t, inv_freq)
    cos = freqs.cos() * self.mscale
    sin = freqs.sin() * self.mscale
    cache = torch.cat((cos, sin), dim=-1)
    return cache