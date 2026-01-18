from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.running import Running
def _cast_and_nan_check_input(self, x: Union[float, Tensor], weight: Optional[Union[float, Tensor]]=None) -> Tuple[Tensor, Tensor]:
    """Convert input ``x`` to a tensor and check for Nans."""
    if not isinstance(x, Tensor):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
    if weight is not None and (not isinstance(weight, Tensor)):
        weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
    nans = torch.isnan(x)
    if weight is not None:
        nans_weight = torch.isnan(weight)
    else:
        nans_weight = torch.zeros_like(nans).bool()
        weight = torch.ones_like(x)
    if nans.any() or nans_weight.any():
        if self.nan_strategy == 'error':
            raise RuntimeError('Encountered `nan` values in tensor')
        if self.nan_strategy in ('ignore', 'warn'):
            if self.nan_strategy == 'warn':
                rank_zero_warn('Encountered `nan` values in tensor. Will be removed.', UserWarning)
            x = x[~(nans | nans_weight)]
            weight = weight[~(nans | nans_weight)]
        else:
            if not isinstance(self.nan_strategy, float):
                raise ValueError(f'`nan_strategy` shall be float but you pass {self.nan_strategy}')
            x[nans | nans_weight] = self.nan_strategy
            weight[nans | nans_weight] = self.nan_strategy
    return (x.to(self.dtype), weight.to(self.dtype))