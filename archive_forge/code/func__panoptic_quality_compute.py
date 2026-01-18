from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _panoptic_quality_compute(iou_sum: Tensor, true_positives: Tensor, false_positives: Tensor, false_negatives: Tensor) -> Tensor:
    """Compute the final panoptic quality from interim values.

    Args:
        iou_sum: the iou sum from the update step
        true_positives: the TP value from the update step
        false_positives: the FP value from the update step
        false_negatives: the FN value from the update step

    Returns:
        Panoptic quality as a tensor containing a single scalar.

    """
    denominator = (true_positives + 0.5 * false_positives + 0.5 * false_negatives).double()
    panoptic_quality = torch.where(denominator > 0.0, iou_sum / denominator, 0.0)
    return torch.mean(panoptic_quality[denominator > 0])