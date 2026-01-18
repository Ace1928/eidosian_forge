from enum import Enum
from typing import Optional, Tuple
import torch
from torch import Tensor
def _is_sparsity_zero(dense: Tensor, topk_percent: Optional[float], topk_element: Optional[int], top_k_dim: Optional[int]) -> bool:
    """Returns True when a given value of topk_percent or topk_element along a particular top_k_dim
    for an input tensor results in sparsity=0% (or top-100-percent). Otherwise, returns False.
    """
    if topk_percent is None and topk_element is None:
        return False
    top_k_total_size = _top_k_total_size(dense, top_k_dim)
    k = _get_k_for_topk(topk_percent, topk_element, top_k_total_size)
    return k == top_k_total_size