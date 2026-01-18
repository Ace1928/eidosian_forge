from typing import Literal, Optional, Sequence, Union
import torch
from torch import Tensor
from torchmetrics.functional.text.helper import _LevenshteinEditDistance as _LE_distance
def _edit_distance_compute(edit_scores: Tensor, num_elements: Union[Tensor, int], reduction: Optional[Literal['mean', 'sum', 'none']]='mean') -> Tensor:
    """Compute final edit distance reduced over the batch."""
    if edit_scores.numel() == 0:
        return torch.tensor(0, dtype=torch.int32)
    if reduction == 'mean':
        return edit_scores.sum() / num_elements
    if reduction == 'sum':
        return edit_scores.sum()
    if reduction is None or reduction == 'none':
        return edit_scores
    raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")