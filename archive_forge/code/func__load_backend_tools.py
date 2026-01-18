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
def _load_backend_tools(backend: Literal['pycocotools', 'faster_coco_eval']) -> Tuple[object, object, ModuleType]:
    """Load the backend tools for the given backend."""
    if backend == 'pycocotools':
        if not _PYCOCOTOOLS_AVAILABLE:
            raise ModuleNotFoundError('Backend `pycocotools` in metric `MeanAveragePrecision`  metric requires that `pycocotools` is installed. Please install with `pip install pycocotools` or `pip install torchmetrics[detection]`')
        import pycocotools.mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        return (COCO, COCOeval, mask_utils)
    if not _FASTER_COCO_EVAL_AVAILABLE:
        raise ModuleNotFoundError('Backend `faster_coco_eval` in metric `MeanAveragePrecision`  metric requires that `faster-coco-eval` is installed. Please install with `pip install faster-coco-eval`.')
    from faster_coco_eval import COCO
    from faster_coco_eval import COCOeval_faster as COCOeval
    from faster_coco_eval.core import mask as mask_utils
    return (COCO, COCOeval, mask_utils)