from typing import Optional, Tuple
import torch
from torch import Tensor
def _safe_xlogy(x: Tensor, y: Tensor) -> Tensor:
    """Compute x * log(y). Returns 0 if x=0.

    Example:
        >>> import torch
        >>> x = torch.zeros(1)
        >>> _safe_xlogy(x, 1/x)
        tensor([0.])

    """
    res = x * torch.log(y)
    res[x == 0] = 0.0
    return res