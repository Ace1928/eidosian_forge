from typing import Optional
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix
from torchmetrics.utilities.compute import _safe_matmul
def _pairwise_cosine_similarity_update(x: Tensor, y: Optional[Tensor]=None, zero_diagonal: Optional[bool]=None) -> Tensor:
    """Calculate the pairwise cosine similarity matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
    norm = torch.norm(x, p=2, dim=1)
    x = x / norm.unsqueeze(1)
    norm = torch.norm(y, p=2, dim=1)
    y = y / norm.unsqueeze(1)
    distance = _safe_matmul(x, y)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance