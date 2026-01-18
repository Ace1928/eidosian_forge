from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix
def _pairwise_manhattan_distance_update(x: Tensor, y: Optional[Tensor]=None, zero_diagonal: Optional[bool]=None) -> Tensor:
    """Calculate the pairwise manhattan similarity matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: if provided, a tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero

    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
    distance = (x.unsqueeze(1) - y.unsqueeze(0).repeat(x.shape[0], 1, 1)).abs().sum(dim=-1)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance