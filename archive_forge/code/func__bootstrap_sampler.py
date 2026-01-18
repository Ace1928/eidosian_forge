from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric
def _bootstrap_sampler(size: int, sampling_strategy: str='poisson') -> Tensor:
    """Resample a tensor along its first dimension with replacement.

    Args:
        size: number of samples
        sampling_strategy: the strategy to use for sampling, either ``'poisson'`` or ``'multinomial'``

    Returns:
        resampled tensor

    """
    if sampling_strategy == 'poisson':
        p = torch.distributions.Poisson(1)
        n = p.sample((size,))
        return torch.arange(size).repeat_interleave(n.long(), dim=0)
    if sampling_strategy == 'multinomial':
        return torch.multinomial(torch.ones(size), num_samples=size, replacement=True)
    raise ValueError('Unknown sampling strategy')