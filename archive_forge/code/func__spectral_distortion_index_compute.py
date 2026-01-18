from typing import Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.utilities.distributed import reduce
def _spectral_distortion_index_compute(preds: Tensor, target: Tensor, p: int=1, reduction: Literal['elementwise_mean', 'sum', 'none']='elementwise_mean') -> Tensor:
    """Compute Spectral Distortion Index (SpectralDistortionIndex_).

    Args:
        preds: Low resolution multispectral image
        target: High resolution fused image
        p: a parameter to emphasize large spectral difference
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Example:
        >>> _ = torch.manual_seed(42)
        >>> preds = torch.rand([16, 3, 16, 16])
        >>> target = torch.rand([16, 3, 16, 16])
        >>> preds, target = _spectral_distortion_index_update(preds, target)
        >>> _spectral_distortion_index_compute(preds, target)
        tensor(0.0234)

    """
    length = preds.shape[1]
    m1 = torch.zeros((length, length), device=preds.device)
    m2 = torch.zeros((length, length), device=preds.device)
    for k in range(length):
        num = length - (k + 1)
        if num == 0:
            continue
        stack1 = target[:, k:k + 1, :, :].repeat(num, 1, 1, 1)
        stack2 = torch.cat([target[:, r:r + 1, :, :] for r in range(k + 1, length)], dim=0)
        score = [s.mean() for s in universal_image_quality_index(stack1, stack2, reduction='none').split(preds.shape[0])]
        m1[k, k + 1:] = torch.stack(score, 0)
        stack1 = preds[:, k:k + 1, :, :].repeat(num, 1, 1, 1)
        stack2 = torch.cat([preds[:, r:r + 1, :, :] for r in range(k + 1, length)], dim=0)
        score = [s.mean() for s in universal_image_quality_index(stack1, stack2, reduction='none').split(preds.shape[0])]
        m2[k, k + 1:] = torch.stack(score, 0)
    m1 = m1 + m1.T
    m2 = m2 + m2.T
    diff = torch.pow(torch.abs(m1 - m2), p)
    if length == 1:
        output = torch.pow(diff, 1.0 / p)
    else:
        output = torch.pow(1.0 / (length * (length - 1)) * torch.sum(diff), 1.0 / p)
    return reduce(output, reduction)