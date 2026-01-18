from typing import Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.enums import ClassificationTaskNoBinary
Compute Exact match (also known as subset accuracy).

    Exact Match is a stricter version of accuracy where all classes/labels have to match exactly for the sample to be
    correctly classified.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'multiclass'`` or ``'multilabel'``. See the documentation of
    :func:`~torchmetrics.functional.classification.multiclass_exact_match` and
    :func:`~torchmetrics.functional.classification.multilabel_exact_match` for the specific details of
    each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> exact_match(preds, target, task="multiclass", num_classes=3, multidim_average='global')
        tensor(0.5000)

        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 1], [2, 1], [0, 2]], [[2, 2], [2, 1], [1, 0]]])
        >>> exact_match(preds, target, task="multiclass", num_classes=3, multidim_average='samplewise')
        tensor([1., 0.])

    