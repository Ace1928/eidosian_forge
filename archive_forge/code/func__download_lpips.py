from typing import Any, ClassVar, List, Optional, Sequence, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.lpips import _LPIPS, _lpips_compute, _lpips_update, _NoTrainLpips
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _download_lpips() -> None:
    _LPIPS(pretrained=True, net='vgg')