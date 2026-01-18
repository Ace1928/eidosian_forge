from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric
def _get_args_kwargs_by_output(self, *args: Tensor, **kwargs: Tensor) -> List[Tuple[Tensor, Tensor]]:
    """Get args and kwargs reshaped to be output-specific and (maybe) having NaNs stripped out."""
    args_kwargs_by_output = []
    for i in range(len(self.metrics)):
        selected_args = apply_to_collection(args, Tensor, torch.index_select, dim=self.output_dim, index=torch.tensor(i, device=self.device))
        selected_kwargs = apply_to_collection(kwargs, Tensor, torch.index_select, dim=self.output_dim, index=torch.tensor(i, device=self.device))
        if self.remove_nans:
            args_kwargs = selected_args + tuple(selected_kwargs.values())
            nan_idxs = _get_nan_indices(*args_kwargs)
            selected_args = [arg[~nan_idxs] for arg in selected_args]
            selected_kwargs = {k: v[~nan_idxs] for k, v in selected_kwargs.items()}
        if self.squeeze_outputs:
            selected_args = [arg.squeeze(self.output_dim) for arg in selected_args]
            selected_kwargs = {k: v.squeeze(self.output_dim) for k, v in selected_kwargs.items()}
        args_kwargs_by_output.append((selected_args, selected_kwargs))
    return args_kwargs_by_output