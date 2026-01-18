from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _drop_negative_ignored_indices(preds: Tensor, target: Tensor, ignore_index: int, mode: DataType) -> Tuple[Tensor, Tensor]:
    """Remove negative ignored indices.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        ignore_index: Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
        mode: Mode of the input tensors

    Return:
        Tensors of preds and target without negative ignore target values.

    """
    if mode == mode.MULTIDIM_MULTICLASS and preds.dtype == torch.float:
        num_dims = len(preds.shape)
        num_classes = preds.shape[1]
        preds = preds.transpose(1, num_dims - 1)
        preds = preds.reshape(-1, num_classes)
        target = target.reshape(-1)
    if mode in [mode.MULTICLASS, mode.MULTIDIM_MULTICLASS]:
        preds = preds[target != ignore_index]
        target = target[target != ignore_index]
    return (preds, target)