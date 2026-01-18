from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
def _cohen_kappa_reduce(confmat: Tensor, weights: Optional[Literal['linear', 'quadratic', 'none']]=None) -> Tensor:
    """Reduce an un-normalized confusion matrix of shape (n_classes, n_classes) into the cohen kappa score."""
    confmat = confmat.float() if not confmat.is_floating_point() else confmat
    num_classes = confmat.shape[0]
    sum0 = confmat.sum(dim=0, keepdim=True)
    sum1 = confmat.sum(dim=1, keepdim=True)
    expected = sum1 @ sum0 / sum0.sum()
    if weights is None or weights == 'none':
        w_mat = torch.ones_like(confmat).flatten()
        w_mat[::num_classes + 1] = 0
        w_mat = w_mat.reshape(num_classes, num_classes)
    elif weights in ('linear', 'quadratic'):
        w_mat = torch.zeros_like(confmat)
        w_mat += torch.arange(num_classes, dtype=w_mat.dtype, device=w_mat.device)
        w_mat = torch.abs(w_mat - w_mat.T) if weights == 'linear' else torch.pow(w_mat - w_mat.T, 2.0)
    else:
        raise ValueError(f"Received {weights} for argument ``weights`` but should be either None, 'linear' or 'quadratic'")
    k = torch.sum(w_mat * confmat) / torch.sum(w_mat * expected)
    return 1 - k