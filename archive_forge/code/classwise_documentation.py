import typing
from typing import Any, Dict, List, Optional, Sequence, Union
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.abstract import WrapperMetric
Set attribute to classwise wrapper.