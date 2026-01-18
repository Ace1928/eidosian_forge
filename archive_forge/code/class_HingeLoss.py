from typing import Any, Optional, Sequence, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.hinge import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class HingeLoss(_ClassificationTaskWrapper):
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :class:`~torchmetrics.classification.BinaryHingeLoss` and :class:`~torchmetrics.classification.MulticlassHingeLoss`
    for the specific details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> target = tensor([0, 1, 1])
        >>> preds = tensor([0.5, 0.7, 0.1])
        >>> hinge = HingeLoss(task="binary")
        >>> hinge(preds, target)
        tensor(0.9000)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge = HingeLoss(task="multiclass", num_classes=3)
        >>> hinge(preds, target)
        tensor(1.5551)

        >>> target = tensor([0, 1, 2])
        >>> preds = tensor([[-1.0, 0.9, 0.2], [0.5, -1.1, 0.8], [2.2, -0.5, 0.3]])
        >>> hinge = HingeLoss(task="multiclass", num_classes=3, multiclass_mode="one-vs-all")
        >>> hinge(preds, target)
        tensor([1.3743, 1.1945, 1.2359])

    """

    def __new__(cls: Type['HingeLoss'], task: Literal['binary', 'multiclass'], num_classes: Optional[int]=None, squared: bool=False, multiclass_mode: Optional[Literal['crammer-singer', 'one-vs-all']]='crammer-singer', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update({'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryHingeLoss(squared, **kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            if multiclass_mode not in ('crammer-singer', 'one-vs-all'):
                raise ValueError(f"`multiclass_mode` is expected to be one of 'crammer-singer' or 'one-vs-all' but `{multiclass_mode}` was passed.")
            return MulticlassHingeLoss(num_classes, squared, multiclass_mode, **kwargs)
        raise ValueError(f'Unsupported task `{task}`')