from typing import Optional, Tuple
import torch
from torch import Tensor
def _adjust_weights_safe_divide(score: Tensor, average: Optional[str], multilabel: bool, tp: Tensor, fp: Tensor, fn: Tensor, top_k: int=1) -> Tensor:
    if average is None or average == 'none':
        return score
    if average == 'weighted':
        weights = tp + fn
    else:
        weights = torch.ones_like(score)
        if not multilabel:
            weights[tp + fp + fn == 0 if top_k == 1 else tp + fn == 0] = 0.0
    return _safe_divide(weights * score, weights.sum(-1, keepdim=True)).sum(-1)