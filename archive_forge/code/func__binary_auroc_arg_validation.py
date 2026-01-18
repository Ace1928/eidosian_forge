from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.functional.classification.roc import (
from torchmetrics.utilities.compute import _auc_compute_without_check, _safe_divide
from torchmetrics.utilities.data import _bincount
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.prints import rank_zero_warn
def _binary_auroc_arg_validation(max_fpr: Optional[float]=None, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None) -> None:
    _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
    if max_fpr is not None and (not isinstance(max_fpr, float)) and (0 < max_fpr <= 1):
        raise ValueError(f'Arguments `max_fpr` should be a float in range (0, 1], but got: {max_fpr}')