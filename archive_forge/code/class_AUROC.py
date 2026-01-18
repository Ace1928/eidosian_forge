from typing import Any, List, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.precision_recall_curve import (
from torchmetrics.functional.classification.auroc import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class AUROC(_ClassificationTaskWrapper):
    """Compute Area Under the Receiver Operating Characteristic Curve (`ROC AUC`_).

    The AUROC score summarizes the ROC curve into an single number that describes the performance of a model for
    multiple thresholds at the same time. Notably, an AUROC score of 1 is a perfect score and an AUROC score of 0.5
    corresponds to random guessing.

    This module is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryAUROC`, :class:`~torchmetrics.classification.MulticlassAUROC` and
    :class:`~torchmetrics.classification.MultilabelAUROC` for the specific details of each argument influence and
    examples.

    Legacy Example:
        >>> from torch import tensor
        >>> preds = tensor([0.13, 0.26, 0.08, 0.19, 0.34])
        >>> target = tensor([0, 0, 1, 1, 1])
        >>> auroc = AUROC(task="binary")
        >>> auroc(preds, target)
        tensor(0.5000)

        >>> preds = tensor([[0.90, 0.05, 0.05],
        ...                       [0.05, 0.90, 0.05],
        ...                       [0.05, 0.05, 0.90],
        ...                       [0.85, 0.05, 0.10],
        ...                       [0.10, 0.10, 0.80]])
        >>> target = tensor([0, 1, 1, 2, 2])
        >>> auroc = AUROC(task="multiclass", num_classes=3)
        >>> auroc(preds, target)
        tensor(0.7778)

    """

    def __new__(cls: Type['AUROC'], task: Literal['binary', 'multiclass', 'multilabel'], thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, average: Optional[Literal['macro', 'weighted', 'none']]='macro', max_fpr: Optional[float]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({'thresholds': thresholds, 'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryAUROC(max_fpr, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            return MulticlassAUROC(num_classes, average, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
            return MultilabelAUROC(num_labels, average, **kwargs)
        raise ValueError(f'Task {task} not supported!')

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update metric state."""
        raise NotImplementedError(f'{self.__class__.__name__} metric does not have a global `update` method. Use the task specific metric.')

    def compute(self) -> None:
        """Compute metric."""
        raise NotImplementedError(f'{self.__class__.__name__} metric does not have a global `compute` method. Use the task specific metric.')