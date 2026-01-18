from typing import Any, List, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.precision_recall_curve import (
from torchmetrics.functional.classification.average_precision import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class AveragePrecision(_ClassificationTaskWrapper):
    """Compute the average precision (AP) score.

    The AP score summarizes a precision-recall curve as an weighted mean of precisions at each threshold, with the
    difference in recall from the previous threshold as weight:

    .. math::
        AP = \\sum_{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryAveragePrecision`,
    :class:`~torchmetrics.classification.MulticlassAveragePrecision` and
    :class:`~torchmetrics.classification.MultilabelAveragePrecision` for the specific details of each argument
    influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> pred = tensor([0, 0.1, 0.8, 0.4])
        >>> target = tensor([0, 1, 1, 1])
        >>> average_precision = AveragePrecision(task="binary")
        >>> average_precision(pred, target)
        tensor(1.)

        >>> pred = tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = tensor([0, 1, 3, 2])
        >>> average_precision = AveragePrecision(task="multiclass", num_classes=5, average=None)
        >>> average_precision(pred, target)
        tensor([1.0000, 1.0000, 0.2500, 0.2500,    nan])

    """

    def __new__(cls: Type['AveragePrecision'], task: Literal['binary', 'multiclass', 'multilabel'], thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, average: Optional[Literal['macro', 'weighted', 'none']]='macro', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({'thresholds': thresholds, 'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryAveragePrecision(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            return MulticlassAveragePrecision(num_classes, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
            return MultilabelAveragePrecision(num_labels, average, **kwargs)
        raise ValueError(f'Task {task} not supported!')