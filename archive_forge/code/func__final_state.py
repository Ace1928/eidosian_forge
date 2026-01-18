from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
def _final_state(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Aggregate states that are lists and return final states."""
    tp = dim_zero_cat(self.tp)
    fp = dim_zero_cat(self.fp)
    tn = dim_zero_cat(self.tn)
    fn = dim_zero_cat(self.fn)
    return (tp, fp, tn, fn)