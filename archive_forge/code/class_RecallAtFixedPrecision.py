from typing import Any, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.precision_recall_curve import (
from torchmetrics.functional.classification.recall_fixed_precision import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class RecallAtFixedPrecision(_ClassificationTaskWrapper):
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for
    a given precision level.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :class:`~torchmetrics.classification.BinaryRecallAtFixedPrecision`,
    :class:`~torchmetrics.classification.MulticlassRecallAtFixedPrecision` and
    :class:`~torchmetrics.classification.MultilabelRecallAtFixedPrecision` for the specific details of each argument
    influence and examples.

    """

    def __new__(cls: Type['RecallAtFixedPrecision'], task: Literal['binary', 'multiclass', 'multilabel'], min_precision: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, num_classes: Optional[int]=None, num_labels: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        if task == ClassificationTask.BINARY:
            return BinaryRecallAtFixedPrecision(min_precision, thresholds, ignore_index, validate_args, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            return MulticlassRecallAtFixedPrecision(num_classes, min_precision, thresholds, ignore_index, validate_args, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f'`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`')
            return MultilabelRecallAtFixedPrecision(num_labels, min_precision, thresholds, ignore_index, validate_args, **kwargs)
        raise ValueError(f'Task {task} not supported!')