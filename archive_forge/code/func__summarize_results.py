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
def _summarize_results(self, precisions: Tensor, recalls: Tensor) -> Tuple[MAPMetricResults, MARMetricResults]:
    """Summarizes the precision and recall values to calculate mAP/mAR.

        Args:
            precisions:
                Precision values for different thresholds
            recalls:
                Recall values for different thresholds

        """
    results = {'precision': precisions, 'recall': recalls}
    map_metrics = MAPMetricResults()
    last_max_det_thr = self.max_detection_thresholds[-1]
    map_metrics.map = self._summarize(results, True, max_dets=last_max_det_thr)
    if 0.5 in self.iou_thresholds:
        map_metrics.map_50 = self._summarize(results, True, iou_threshold=0.5, max_dets=last_max_det_thr)
    else:
        map_metrics.map_50 = torch.tensor([-1])
    if 0.75 in self.iou_thresholds:
        map_metrics.map_75 = self._summarize(results, True, iou_threshold=0.75, max_dets=last_max_det_thr)
    else:
        map_metrics.map_75 = torch.tensor([-1])
    map_metrics.map_small = self._summarize(results, True, area_range='small', max_dets=last_max_det_thr)
    map_metrics.map_medium = self._summarize(results, True, area_range='medium', max_dets=last_max_det_thr)
    map_metrics.map_large = self._summarize(results, True, area_range='large', max_dets=last_max_det_thr)
    mar_metrics = MARMetricResults()
    for max_det in self.max_detection_thresholds:
        mar_metrics[f'mar_{max_det}'] = self._summarize(results, False, max_dets=max_det)
    mar_metrics.mar_small = self._summarize(results, False, area_range='small', max_dets=last_max_det_thr)
    mar_metrics.mar_medium = self._summarize(results, False, area_range='medium', max_dets=last_max_det_thr)
    mar_metrics.mar_large = self._summarize(results, False, area_range='large', max_dets=last_max_det_thr)
    return (map_metrics, mar_metrics)