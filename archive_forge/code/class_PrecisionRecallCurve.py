from typing import Any, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.auroc import _reduce_auroc
from torchmetrics.functional.classification.precision_recall_curve import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.compute import _auc_compute_without_check
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve
class PrecisionRecallCurve(_ClassificationTaskWrapper):
    """Compute the precision-recall curve.

    The curve consist of multiple pairs of precision and recall values evaluated at different thresholds, such that the
    tradeoff between the two values can been seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryPrecisionRecallCurve`,
    :class:`~torchmetrics.classification.MulticlassPrecisionRecallCurve` and
    :class:`~torchmetrics.classification.MultilabelPrecisionRecallCurve` for the specific details of each argument
    influence and examples.

    Legacy Example:
        >>> pred = torch.tensor([0, 0.1, 0.8, 0.4])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> pr_curve = PrecisionRecallCurve(task="binary")
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        tensor([0.5000, 0.6667, 0.5000, 1.0000, 1.0000])
        >>> recall
        tensor([1.0000, 1.0000, 0.5000, 0.5000, 0.0000])
        >>> thresholds
        tensor([0.0000, 0.1000, 0.4000, 0.8000])

        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=5)
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        [tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 1.0000, 1.0000]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 1., 0.]), tensor([1., 1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]),
         tensor(0.0500)]

    """

    def __new__(cls: Type['PrecisionRecallCurve'], task: Literal['binary', 'multiclass', 'multilabel'], thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({'thresholds': thresholds, 'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryPrecisionRecallCurve(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            return MulticlassPrecisionRecallCurve(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
            return MultilabelPrecisionRecallCurve(num_labels, **kwargs)
        raise ValueError(f'Task {task} not supported!')