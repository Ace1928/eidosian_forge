from typing import Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_xlogy
def _kld_update(p: Tensor, q: Tensor, log_prob: bool) -> Tuple[Tensor, int]:
    """Update and returns KL divergence scores for each observation and the total number of observations.

    Args:
        p: data distribution with shape ``[N, d]``
        q: prior or approximate distribution with shape ``[N, d]``
        log_prob: bool indicating if input is log-probabilities or probabilities. If given as probabilities,
            will normalize to make sure the distributes sum to 1

    """
    _check_same_shape(p, q)
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError(f'Expected both p and q distribution to be 2D but got {p.ndim} and {q.ndim} respectively')
    total = p.shape[0]
    if log_prob:
        measures = torch.sum(p.exp() * (p - q), axis=-1)
    else:
        p = p / p.sum(axis=-1, keepdim=True)
        q = q / q.sum(axis=-1, keepdim=True)
        measures = _safe_xlogy(p, p / q).sum(axis=-1)
    return (measures, total)