from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _panoptic_quality_update_sample(flatten_preds: Tensor, flatten_target: Tensor, cat_id_to_continuous_id: Dict[int, int], void_color: Tuple[int, int], stuffs_modified_metric: Optional[Set[int]]=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate stat scores required to compute the metric **for a single sample**.

    Computed scores: iou sum, true positives, false positives, false negatives.

    NOTE: For the modified PQ case, this implementation uses the `true_positives` output tensor to aggregate the actual
        TPs for things classes, but the number of target segments for stuff classes.
        The `iou_sum` output tensor, instead, aggregates the IoU values at different thresholds (i.e., 0.5 for things
        and 0 for stuffs).
        This allows seamlessly using the same `.compute()` method for both PQ variants.

    Args:
        flatten_preds: A flattened prediction tensor referring to a single sample, shape (num_points, 2).
        flatten_target: A flattened target tensor referring to a single sample, shape (num_points, 2).
        cat_id_to_continuous_id: Mapping from original category IDs to continuous IDs
        void_color: an additional, unused color.
        stuffs_modified_metric: Set of stuff category IDs for which the PQ metric is computed using the "modified"
            formula. If not specified, the original formula is used for all categories.

    Returns:
        - IOU Sum
        - True positives
        - False positives
        - False negatives.

    """
    stuffs_modified_metric = stuffs_modified_metric or set()
    device = flatten_preds.device
    num_categories = len(cat_id_to_continuous_id)
    iou_sum = torch.zeros(num_categories, dtype=torch.double, device=device)
    true_positives = torch.zeros(num_categories, dtype=torch.int, device=device)
    false_positives = torch.zeros(num_categories, dtype=torch.int, device=device)
    false_negatives = torch.zeros(num_categories, dtype=torch.int, device=device)
    pred_areas = cast(Dict[_Color, Tensor], _get_color_areas(flatten_preds))
    target_areas = cast(Dict[_Color, Tensor], _get_color_areas(flatten_target))
    intersection_matrix = torch.transpose(torch.stack((flatten_preds, flatten_target), -1), -1, -2)
    intersection_areas = cast(Dict[Tuple[_Color, _Color], Tensor], _get_color_areas(intersection_matrix))
    pred_segment_matched = set()
    target_segment_matched = set()
    for pred_color, target_color in intersection_areas:
        if target_color == void_color:
            continue
        if pred_color[0] != target_color[0]:
            continue
        iou = _calculate_iou(pred_color, target_color, pred_areas, target_areas, intersection_areas, void_color)
        continuous_id = cat_id_to_continuous_id[target_color[0]]
        if target_color[0] not in stuffs_modified_metric and iou > 0.5:
            pred_segment_matched.add(pred_color)
            target_segment_matched.add(target_color)
            iou_sum[continuous_id] += iou
            true_positives[continuous_id] += 1
        elif target_color[0] in stuffs_modified_metric and iou > 0:
            iou_sum[continuous_id] += iou
    for cat_id in _filter_false_negatives(target_areas, target_segment_matched, intersection_areas, void_color):
        if cat_id not in stuffs_modified_metric:
            continuous_id = cat_id_to_continuous_id[cat_id]
            false_negatives[continuous_id] += 1
    for cat_id in _filter_false_positives(pred_areas, pred_segment_matched, intersection_areas, void_color):
        if cat_id not in stuffs_modified_metric:
            continuous_id = cat_id_to_continuous_id[cat_id]
            false_positives[continuous_id] += 1
    for cat_id, _ in target_areas:
        if cat_id in stuffs_modified_metric:
            continuous_id = cat_id_to_continuous_id[cat_id]
            true_positives[continuous_id] += 1
    return (iou_sum, true_positives, false_positives, false_negatives)