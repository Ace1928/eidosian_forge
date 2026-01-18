import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import IntTensor, Tensor
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _PYCOCOTOOLS_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_8
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _evaluate_image(self, idx: int, class_id: int, area_range: Tuple[int, int], max_det: int, ious: dict) -> Optional[dict]:
    """Perform evaluation for single class and image.

        Args:
            idx:
                Image Id, equivalent to the index of supplied samples.
            class_id:
                Class Id of the supplied ground truth and detection labels.
            area_range:
                List of lower and upper bounding box area threshold.
            max_det:
                Maximum number of evaluated detection bounding boxes.
            ious:
                IoU results for image and class.

        """
    gt = self.groundtruths[idx]
    det = self.detections[idx]
    gt_label_mask = (self.groundtruth_labels[idx] == class_id).nonzero().squeeze(1)
    det_label_mask = (self.detection_labels[idx] == class_id).nonzero().squeeze(1)
    if len(gt_label_mask) == 0 and len(det_label_mask) == 0:
        return None
    num_iou_thrs = len(self.iou_thresholds)
    if len(gt_label_mask) > 0 and len(det_label_mask) == 0:
        return self.__evaluate_image_gt_no_preds(gt, gt_label_mask, area_range, num_iou_thrs)
    if len(gt_label_mask) == 0 and len(det_label_mask) > 0:
        return self.__evaluate_image_preds_no_gt(det, idx, det_label_mask, max_det, area_range, num_iou_thrs)
    gt = [gt[i] for i in gt_label_mask]
    det = [det[i] for i in det_label_mask]
    if len(gt) == 0 and len(det) == 0:
        return None
    if isinstance(det, dict):
        det = [det]
    if isinstance(gt, dict):
        gt = [gt]
    areas = compute_area(gt, iou_type=self.iou_type).to(self.device)
    ignore_area = torch.logical_or(areas < area_range[0], areas > area_range[1])
    ignore_area_sorted, gtind = torch.sort(ignore_area.to(torch.uint8))
    ignore_area_sorted = ignore_area_sorted.to(torch.bool).to(self.device)
    gt = [gt[i] for i in gtind]
    scores = self.detection_scores[idx]
    scores_filtered = scores[det_label_mask]
    scores_sorted, dtind = torch.sort(scores_filtered, descending=True)
    det = [det[i] for i in dtind]
    if len(det) > max_det:
        det = det[:max_det]
    ious = ious[idx, class_id][:, gtind] if len(ious[idx, class_id]) > 0 else ious[idx, class_id]
    num_iou_thrs = len(self.iou_thresholds)
    num_gt = len(gt)
    num_det = len(det)
    gt_matches = torch.zeros((num_iou_thrs, num_gt), dtype=torch.bool, device=self.device)
    det_matches = torch.zeros((num_iou_thrs, num_det), dtype=torch.bool, device=self.device)
    gt_ignore = ignore_area_sorted
    det_ignore = torch.zeros((num_iou_thrs, num_det), dtype=torch.bool, device=self.device)
    if torch.numel(ious) > 0:
        for idx_iou, t in enumerate(self.iou_thresholds):
            for idx_det, _ in enumerate(det):
                m = MeanAveragePrecision._find_best_gt_match(t, gt_matches, idx_iou, gt_ignore, ious, idx_det)
                if m == -1:
                    continue
                det_ignore[idx_iou, idx_det] = gt_ignore[m]
                det_matches[idx_iou, idx_det] = 1
                gt_matches[idx_iou, m] = 1
    det_areas = compute_area(det, iou_type=self.iou_type).to(self.device)
    det_ignore_area = (det_areas < area_range[0]) | (det_areas > area_range[1])
    ar = det_ignore_area.reshape((1, num_det))
    det_ignore = torch.logical_or(det_ignore, torch.logical_and(det_matches == 0, torch.repeat_interleave(ar, num_iou_thrs, 0)))
    return {'dtMatches': det_matches.to(self.device), 'gtMatches': gt_matches.to(self.device), 'dtScores': scores_sorted.to(self.device), 'gtIgnore': gt_ignore.to(self.device), 'dtIgnore': det_ignore.to(self.device)}