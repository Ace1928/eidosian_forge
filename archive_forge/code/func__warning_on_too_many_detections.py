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
def _warning_on_too_many_detections(limit: int) -> None:
    rank_zero_warn(f'Encountered more than {limit} detections in a single image. This means that certain detections with the lowest scores will be ignored, that may have an undesirable impact on performance. Please consider adjusting the `max_detection_threshold` to suit your use case. To disable this warning, set attribute class `warn_on_many_detections=False`, after initializing the metric.', UserWarning)