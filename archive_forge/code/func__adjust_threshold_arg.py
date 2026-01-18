from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _adjust_threshold_arg(thresholds: Optional[Union[int, List[float], Tensor]]=None, device: Optional[torch.device]=None) -> Optional[Tensor]:
    """Convert threshold arg for list and int to tensor format."""
    if isinstance(thresholds, int):
        return torch.linspace(0, 1, thresholds, device=device)
    if isinstance(thresholds, list):
        return torch.tensor(thresholds, device=device)
    return thresholds