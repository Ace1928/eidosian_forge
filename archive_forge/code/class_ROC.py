from typing import Any, List, Optional, Tuple, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.precision_recall_curve import (
from torchmetrics.functional.classification.auroc import _reduce_auroc
from torchmetrics.functional.classification.roc import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.compute import _auc_compute_without_check
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve
class ROC(_ClassificationTaskWrapper):
    """Compute the Receiver Operating Characteristic (ROC).

    The curve consist of multiple pairs of true positive rate (TPR) and false positive rate (FPR) values evaluated at
    different thresholds, such that the tradeoff between the two values can be seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryROC`,
    :class:`~torchmetrics.classification.MulticlassROC` and
    :class:`~torchmetrics.classification.MultilabelROC` for the specific details of each argument
    influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> pred = tensor([0.0, 1.0, 2.0, 3.0])
        >>> target = tensor([0, 1, 1, 1])
        >>> roc = ROC(task="binary")
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        tensor([0., 0., 0., 0., 1.])
        >>> tpr
        tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000])
        >>> thresholds
        tensor([1.0000, 0.9526, 0.8808, 0.7311, 0.5000])

        >>> pred = tensor([[0.75, 0.05, 0.05, 0.05],
        ...                [0.05, 0.75, 0.05, 0.05],
        ...                [0.05, 0.05, 0.75, 0.05],
        ...                [0.05, 0.05, 0.05, 0.75]])
        >>> target = tensor([0, 1, 3, 2])
        >>> roc = ROC(task="multiclass", num_classes=4)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        [tensor([0., 0., 1.]), tensor([0., 0., 1.]), tensor([0.0000, 0.3333, 1.0000]), tensor([0.0000, 0.3333, 1.0000])]
        >>> tpr
        [tensor([0., 1., 1.]), tensor([0., 1., 1.]), tensor([0., 0., 1.]), tensor([0., 0., 1.])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500]),
         tensor([1.0000, 0.7500, 0.0500])]

        >>> pred = tensor([[0.8191, 0.3680, 0.1138],
        ...                [0.3584, 0.7576, 0.1183],
        ...                [0.2286, 0.3468, 0.1338],
        ...                [0.8603, 0.0745, 0.1837]])
        >>> target = tensor([[1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 1, 1]])
        >>> roc = ROC(task='multilabel', num_labels=3)
        >>> fpr, tpr, thresholds = roc(pred, target)
        >>> fpr
        [tensor([0.0000, 0.3333, 0.3333, 0.6667, 1.0000]),
         tensor([0., 0., 0., 1., 1.]),
         tensor([0.0000, 0.0000, 0.3333, 0.6667, 1.0000])]
        >>> tpr
        [tensor([0., 0., 1., 1., 1.]),
         tensor([0.0000, 0.3333, 0.6667, 0.6667, 1.0000]),
         tensor([0., 1., 1., 1., 1.])]
        >>> thresholds
        [tensor([1.0000, 0.8603, 0.8191, 0.3584, 0.2286]),
         tensor([1.0000, 0.7576, 0.3680, 0.3468, 0.0745]),
         tensor([1.0000, 0.1837, 0.1338, 0.1183, 0.1138])]

    """

    def __new__(cls: Type['ROC'], task: Literal['binary', 'multiclass', 'multilabel'], thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({'thresholds': thresholds, 'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryROC(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            return MulticlassROC(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
            return MultilabelROC(num_labels, **kwargs)
        raise ValueError(f'Task {task} not supported!')