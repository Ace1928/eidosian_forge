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
class MAPMetricResults(BaseMetricResults):
    """Class to wrap the final mAP results."""
    __slots__ = ('map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large', 'classes')