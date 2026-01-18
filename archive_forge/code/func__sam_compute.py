from typing import Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _sam_compute(preds: Tensor, target: Tensor, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean') -> Tensor:
    """Compute Spectral Angle Mapper.

    Args:
        preds: estimated image
        target: ground truth image
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

    Example:
        >>> gen = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 16, 16], generator=gen)
        >>> target = torch.rand([16, 3, 16, 16], generator=gen)
        >>> preds, target = _sam_update(preds, target)
        >>> _sam_compute(preds, target)
        tensor(0.5914)

    """
    dot_product = (preds * target).sum(dim=1)
    preds_norm = preds.norm(dim=1)
    target_norm = target.norm(dim=1)
    sam_score = torch.clamp(dot_product / (preds_norm * target_norm), -1, 1).acos()
    return reduce(sam_score, reduction)