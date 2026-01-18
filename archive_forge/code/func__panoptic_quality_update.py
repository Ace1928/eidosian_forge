from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _panoptic_quality_update(flatten_preds: Tensor, flatten_target: Tensor, cat_id_to_continuous_id: Dict[int, int], void_color: Tuple[int, int], modified_metric_stuffs: Optional[Set[int]]=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate stat scores required to compute the metric for a full batch.

    Computed scores: iou sum, true positives, false positives, false negatives.

    Args:
        flatten_preds: A flattened prediction tensor, shape (B, num_points, 2).
        flatten_target: A flattened target tensor, shape (B, num_points, 2).
        cat_id_to_continuous_id: Mapping from original category IDs to continuous IDs.
        void_color: an additional, unused color.
        modified_metric_stuffs: Set of stuff category IDs for which the PQ metric is computed using the "modified"
            formula. If not specified, the original formula is used for all categories.

    Returns:
        - IOU Sum
        - True positives
        - False positives
        - False negatives

    """
    device = flatten_preds.device
    num_categories = len(cat_id_to_continuous_id)
    iou_sum = torch.zeros(num_categories, dtype=torch.double, device=device)
    true_positives = torch.zeros(num_categories, dtype=torch.int, device=device)
    false_positives = torch.zeros(num_categories, dtype=torch.int, device=device)
    false_negatives = torch.zeros(num_categories, dtype=torch.int, device=device)
    for flatten_preds_single, flatten_target_single in zip(flatten_preds, flatten_target):
        result = _panoptic_quality_update_sample(flatten_preds_single, flatten_target_single, cat_id_to_continuous_id, void_color, stuffs_modified_metric=modified_metric_stuffs)
        iou_sum += result[0]
        true_positives += result[1]
        false_positives += result[2]
        false_negatives += result[3]
    return (iou_sum, true_positives, false_positives, false_negatives)