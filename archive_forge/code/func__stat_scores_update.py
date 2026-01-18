from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _stat_scores_update(preds: Tensor, target: Tensor, reduce: Optional[str]='micro', mdmc_reduce: Optional[str]=None, num_classes: Optional[int]=None, top_k: Optional[int]=1, threshold: float=0.5, multiclass: Optional[bool]=None, ignore_index: Optional[int]=None, mode: Optional[DataType]=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate true positives, false positives, true negatives, false negatives.

    Raises:
        ValueError:
            The `ignore_index` is not valid
        ValueError:
            When `ignore_index` is used with binary data
        ValueError:
            When inputs are multi-dimensional multi-class, and the ``mdmc_reduce`` parameter is not set

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduce: Defines the reduction that is applied
        mdmc_reduce: Defines how the multi-dimensional multi-class inputs are handled
        num_classes: Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.
        top_k: Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities
        multiclass: Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be
        ignore_index: Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
        mode: Mode of the input tensors

    """
    _negative_index_dropped = False
    if ignore_index is not None and ignore_index < 0 and (mode is not None):
        preds, target = _drop_negative_ignored_indices(preds, target, ignore_index, mode)
        _negative_index_dropped = True
    preds, target, _ = _input_format_classification(preds, target, threshold=threshold, num_classes=num_classes, multiclass=multiclass, top_k=top_k, ignore_index=ignore_index)
    if ignore_index is not None and ignore_index >= preds.shape[1]:
        raise ValueError(f'The `ignore_index` {ignore_index} is not valid for inputs with {preds.shape[1]} classes')
    if ignore_index is not None and preds.shape[1] == 1:
        raise ValueError('You can not use `ignore_index` with binary data.')
    if preds.ndim == 3:
        if not mdmc_reduce:
            raise ValueError('When your inputs are multi-dimensional multi-class, you have to set the `mdmc_reduce` parameter')
        if mdmc_reduce == 'global':
            preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
            target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])
    if ignore_index is not None and reduce != 'macro' and (not _negative_index_dropped):
        preds = _del_column(preds, ignore_index)
        target = _del_column(target, ignore_index)
    tp, fp, tn, fn = _stat_scores(preds, target, reduce=reduce)
    if ignore_index is not None and reduce == 'macro' and (not _negative_index_dropped):
        tp[..., ignore_index] = -1
        fp[..., ignore_index] = -1
        tn[..., ignore_index] = -1
        fn[..., ignore_index] = -1
    return (tp, fp, tn, fn)