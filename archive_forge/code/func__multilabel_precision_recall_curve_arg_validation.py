from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide, interp
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask
def _multilabel_precision_recall_curve_arg_validation(num_labels: int, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None) -> None:
    """Validate non tensor input.

    - ``num_labels`` has to be an int larger than 1
    - ``threshold`` has to be None | a 1d tensor | a list of floats in the [0,1] range | an int
    - ``ignore_index`` has to be None or int

    """
    _multiclass_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)