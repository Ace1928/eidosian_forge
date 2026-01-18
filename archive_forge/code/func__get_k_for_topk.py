from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _get_k_for_topk(topk_percent: Optional[float], top_k_element: Optional[int], top_k_total_size: int) -> int:
    """Converts the top_k_percent to top_k_element when top_k_percent is provided
    as the criterion for top-k calculation. When, top_k_element is used as the criterion,
    simply returns the value for k. Also, ensures k is never 0 to avoid all-zero tensors.
    """
    if top_k_element is None:
        top_k_element = round(top_k_total_size * topk_percent / 100.0)
    elif top_k_element > top_k_total_size:
        raise ValueError('top_k_element for sst or dst is larger than max number of elements along top_k_dim')
    return max(1, top_k_element)