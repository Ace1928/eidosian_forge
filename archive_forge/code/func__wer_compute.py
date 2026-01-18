from typing import List, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _edit_distance
def _wer_compute(errors: Tensor, total: Tensor) -> Tensor:
    """Compute the word error rate.

    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of words overall references

    Returns:
        Word error rate score

    """
    return errors / total