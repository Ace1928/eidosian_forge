from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod
def _multiclass_stat_scores_update(preds: Tensor, target: Tensor, num_classes: int, top_k: int=1, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the statistics.

    - If ``multidim_average`` is equal to samplewise or ``top_k`` is not 1, we transform both preds and
    target into one hot format.
    - Else we calculate statistics by first calculating the confusion matrix and afterwards deriving the
    statistics from that
    - Remove all datapoints that should be ignored. Depending on if ``ignore_index`` is in the set of labels
    or outside we have do use different augmentation strategies when one hot encoding.

    """
    if multidim_average == 'samplewise' or top_k != 1:
        ignore_in = 0 <= ignore_index <= num_classes - 1 if ignore_index is not None else None
        if ignore_index is not None and (not ignore_in):
            preds = preds.clone()
            target = target.clone()
            idx = target == ignore_index
            target[idx] = num_classes
            idx = idx.unsqueeze(1).repeat(1, num_classes, 1) if preds.ndim > target.ndim else idx
            preds[idx] = num_classes
        if top_k > 1:
            preds_oh = torch.movedim(select_topk(preds, topk=top_k, dim=1), 1, -1)
        else:
            preds_oh = torch.nn.functional.one_hot(preds.long(), num_classes + 1 if ignore_index is not None and (not ignore_in) else num_classes)
        target_oh = torch.nn.functional.one_hot(target.long(), num_classes + 1 if ignore_index is not None and (not ignore_in) else num_classes)
        if ignore_index is not None:
            if 0 <= ignore_index <= num_classes - 1:
                target_oh[target == ignore_index, :] = -1
            else:
                preds_oh = preds_oh[..., :-1] if top_k == 1 else preds_oh
                target_oh = target_oh[..., :-1]
                target_oh[target == num_classes, :] = -1
        sum_dim = [0, 1] if multidim_average == 'global' else [1]
        tp = ((target_oh == preds_oh) & (target_oh == 1)).sum(sum_dim)
        fn = ((target_oh != preds_oh) & (target_oh == 1)).sum(sum_dim)
        fp = ((target_oh != preds_oh) & (target_oh == 0)).sum(sum_dim)
        tn = ((target_oh == preds_oh) & (target_oh == 0)).sum(sum_dim)
    elif average == 'micro':
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        tp = (preds == target).sum()
        fp = (preds != target).sum()
        fn = (preds != target).sum()
        tn = num_classes * preds.numel() - (fp + fn + tp)
    else:
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        unique_mapping = target.to(torch.long) * num_classes + preds.to(torch.long)
        bins = _bincount(unique_mapping, minlength=num_classes ** 2)
        confmat = bins.reshape(num_classes, num_classes)
        tp = confmat.diag()
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)
    return (tp, fp, tn, fn)