from typing import Optional
import torch
from torch import Tensor
from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores, _stat_scores_update
from torchmetrics.utilities.checks import _input_squeeze
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod
def _dice_compute(tp: Tensor, fp: Tensor, fn: Tensor, average: Optional[str], mdmc_average: Optional[str], zero_division: int=0) -> Tensor:
    """Compute dice from the stat scores: true positives, false positives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        average: Defines the reduction that is applied
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter)
        zero_division: The value to use for the score if denominator equals zero.
    """
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn
    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = tp + fp + fn == 0
        numerator = numerator[~cond]
        denominator = denominator[~cond]
    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        meaningless_indices = torch.nonzero(tp | fn | fp == 0).cpu()
        numerator[meaningless_indices, ...] = -1
        denominator[meaningless_indices, ...] = -1
    return _reduce_stat_scores(numerator=numerator, denominator=denominator, weights=None if average != 'weighted' else tp + fn, average=average, mdmc_average=mdmc_average, zero_division=zero_division)