from typing import Any, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.confusion_matrix import (
from torchmetrics.functional.classification.jaccard import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class MulticlassJaccardIndex(MulticlassConfusionMatrix):
    """Calculate the Jaccard index for multiclass tasks.

    The `Jaccard index`_ (also known as the intersetion over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \\frac{|A\\cap B|}{|A\\cup B|}

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcji`` (:class:`~torch.Tensor`): A tensor containing the Multi-class Jaccard Index.

    Args:
        num_classes: Integer specifying the number of classes
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassJaccardIndex
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassJaccardIndex(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6667)

    Example (pred is float tensor):
        >>> from torchmetrics.classification import MulticlassJaccardIndex
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> metric = MulticlassJaccardIndex(num_classes=3)
        >>> metric(preds, target)
        tensor(0.6667)

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Class'

    def __init__(self, num_classes: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(num_classes=num_classes, ignore_index=ignore_index, normalize=None, validate_args=False, **kwargs)
        if validate_args:
            _multiclass_jaccard_index_arg_validation(num_classes, ignore_index, average)
        self.validate_args = validate_args
        self.average = average

    def compute(self) -> Tensor:
        """Compute metric."""
        return _jaccard_index_reduce(self.confmat, average=self.average, ignore_index=self.ignore_index)

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]]=None, ax: Optional[_AX_TYPE]=None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value per class
            >>> from torch import randint
            >>> from torchmetrics.classification import MulticlassJaccardIndex
            >>> metric = MulticlassJaccardIndex(num_classes=3, average=None)
            >>> metric.update(randint(3, (20,)), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting a multiple values per class
            >>> from torch import randint
            >>> from torchmetrics.classification import MulticlassJaccardIndex
            >>> metric = MulticlassJaccardIndex(num_classes=3, average=None)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,)), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)