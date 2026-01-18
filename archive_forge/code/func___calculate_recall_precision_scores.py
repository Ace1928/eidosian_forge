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
@staticmethod
def __calculate_recall_precision_scores(recall: Tensor, precision: Tensor, scores: Tensor, idx_cls: int, idx_bbox_area: int, idx_max_det_thrs: int, eval_imgs: list, rec_thresholds: Tensor, max_det: int, num_imgs: int, num_bbox_areas: int) -> Tuple[Tensor, Tensor, Tensor]:
    num_rec_thrs = len(rec_thresholds)
    idx_cls_pointer = idx_cls * num_bbox_areas * num_imgs
    idx_bbox_area_pointer = idx_bbox_area * num_imgs
    img_eval_cls_bbox = [eval_imgs[idx_cls_pointer + idx_bbox_area_pointer + i] for i in range(num_imgs)]
    img_eval_cls_bbox = [e for e in img_eval_cls_bbox if e is not None]
    if not img_eval_cls_bbox:
        return (recall, precision, scores)
    det_scores = torch.cat([e['dtScores'][:max_det] for e in img_eval_cls_bbox])
    dtype = torch.uint8 if det_scores.is_cuda and det_scores.dtype is torch.bool else det_scores.dtype
    inds = torch.argsort(det_scores.to(dtype), descending=True)
    det_scores_sorted = det_scores[inds]
    det_matches = torch.cat([e['dtMatches'][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, inds]
    det_ignore = torch.cat([e['dtIgnore'][:, :max_det] for e in img_eval_cls_bbox], axis=1)[:, inds]
    gt_ignore = torch.cat([e['gtIgnore'] for e in img_eval_cls_bbox])
    npig = torch.count_nonzero(gt_ignore == False)
    if npig == 0:
        return (recall, precision, scores)
    tps = torch.logical_and(det_matches, torch.logical_not(det_ignore))
    fps = torch.logical_and(torch.logical_not(det_matches), torch.logical_not(det_ignore))
    tp_sum = _cumsum(tps, dim=1, dtype=torch.float)
    fp_sum = _cumsum(fps, dim=1, dtype=torch.float)
    for idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
        tp_len = len(tp)
        rc = tp / npig
        pr = tp / (fp + tp + torch.finfo(torch.float64).eps)
        prec = torch.zeros((num_rec_thrs,))
        score = torch.zeros((num_rec_thrs,))
        recall[idx, idx_cls, idx_bbox_area, idx_max_det_thrs] = rc[-1] if tp_len else 0
        diff_zero = torch.zeros((1,), device=pr.device)
        diff = torch.ones((1,), device=pr.device)
        while not torch.all(diff == 0):
            diff = torch.clamp(torch.cat((pr[1:] - pr[:-1], diff_zero), 0), min=0)
            pr += diff
        inds = torch.searchsorted(rc, rec_thresholds.to(rc.device), right=False)
        num_inds = inds.argmax() if inds.max() >= tp_len else num_rec_thrs
        inds = inds[:num_inds]
        prec[:num_inds] = pr[inds]
        score[:num_inds] = det_scores_sorted[inds]
        precision[idx, :, idx_cls, idx_bbox_area, idx_max_det_thrs] = prec
        scores[idx, :, idx_cls, idx_bbox_area, idx_max_det_thrs] = score
    return (recall, precision, scores)