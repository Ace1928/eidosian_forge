import math
from typing import List, Optional
import torch
from torch import nn, Tensor
from .image_list import ImageList
def _generate_wh_pairs(self, num_outputs: int, dtype: torch.dtype=torch.float32, device: torch.device=torch.device('cpu')) -> List[Tensor]:
    _wh_pairs: List[Tensor] = []
    for k in range(num_outputs):
        s_k = self.scales[k]
        s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
        wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]
        for ar in self.aspect_ratios[k]:
            sq_ar = math.sqrt(ar)
            w = self.scales[k] * sq_ar
            h = self.scales[k] / sq_ar
            wh_pairs.extend([[w, h], [h, w]])
        _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
    return _wh_pairs