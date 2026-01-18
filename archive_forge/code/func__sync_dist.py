import contextlib
import io
import json
from types import ModuleType
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from lightning_utilities import apply_to_collection
from torch import Tensor
from torch import distributed as dist
from typing_extensions import Literal
from torchmetrics.detection.helpers import _fix_empty_tensors, _input_validator, _validate_iou_type_arg
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.imports import (
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _sync_dist(self, dist_sync_fn: Optional[Callable]=None, process_group: Optional[Any]=None) -> None:
    """Custom sync function.

        For the iou_type `segm` the detections and groundtruths are no longer tensors but tuples. Therefore, we need
        to gather the list of tuples and then convert it back to a list of tuples.

        """
    super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)
    if 'segm' in self.iou_type:
        self.detection_mask = self._gather_tuple_list(self.detection_mask, process_group)
        self.groundtruth_mask = self._gather_tuple_list(self.groundtruth_mask, process_group)