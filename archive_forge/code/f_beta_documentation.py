from typing import Optional
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _adjust_weights_safe_divide, _safe_divide
from torchmetrics.utilities.enums import ClassificationTask
Compute F-1 score.

    .. math::
        F_{1} = 2\frac{\text{precision} * \text{recall}}{(\text{precision}) + \text{recall}}

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_f1_score`,
    :func:`~torchmetrics.functional.classification.multiclass_f1_score` and
    :func:`~torchmetrics.functional.classification.multilabel_f1_score` for the specific
    details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([0, 1, 2, 0, 1, 2])
        >>> preds = tensor([0, 2, 1, 0, 0, 1])
        >>> f1_score(preds, target, task="multiclass", num_classes=3)
        tensor(0.3333)

    