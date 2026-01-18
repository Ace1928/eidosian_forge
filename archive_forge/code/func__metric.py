from typing import Optional, Sequence, Union
from torch import Tensor
from torchmetrics.functional.retrieval.r_precision import retrieval_r_precision
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
    return retrieval_r_precision(preds, target)