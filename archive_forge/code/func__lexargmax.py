from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.utilities.enums import ClassificationTask
def _lexargmax(x: Tensor) -> Tensor:
    """Returns the index of the maximum value in a list of tuples according to lexicographic ordering.

    Based on https://stackoverflow.com/a/65615160

    """
    idx: Optional[Tensor] = None
    for k in range(x.shape[1]):
        col: Tensor = x[idx, k] if idx is not None else x[:, k]
        z = torch.where(col == col.max())[0]
        idx = z if idx is None else idx[z]
        if len(idx) < 2:
            break
    if idx is None:
        raise ValueError('Failed to extract index')
    return idx