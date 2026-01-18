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
class MultilabelJaccardIndex(MultilabelConfusionMatrix):
    """Calculate the Jaccard index for multilabel tasks.

    The `Jaccard index`_ (also known as the intersetion over union or jaccard similarity coefficient) is an statistic
    that can be used to determine the similarity and diversity of a sample set. It is defined as the size of the
    intersection divided by the union of the sample sets:

    .. math:: J(A,B) = \\frac{|A\\cap B|}{|A\\cup B|}

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A int tensor or float tensor of shape ``(N, C, ...)``. If preds is a
      floating point tensor with values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlji`` (:class:`~torch.Tensor`): A tensor containing the Multi-label Jaccard Index loss.

    Args:
        num_classes: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
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

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MultilabelJaccardIndex
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> metric = MultilabelJaccardIndex(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MultilabelJaccardIndex
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> metric = MultilabelJaccardIndex(num_labels=3)
        >>> metric(preds, target)
        tensor(0.5000)

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Label'

    def __init__(self, num_labels: int, threshold: float=0.5, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(num_labels=num_labels, threshold=threshold, ignore_index=ignore_index, normalize=None, validate_args=False, **kwargs)
        if validate_args:
            _multilabel_jaccard_index_arg_validation(num_labels, threshold, ignore_index, average)
        self.validate_args = validate_args
        self.average = average

    def compute(self) -> Tensor:
        """Compute metric."""
        return _jaccard_index_reduce(self.confmat, average=self.average)

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]]=None, ax: Optional[_AX_TYPE]=None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import MultilabelJaccardIndex
            >>> metric = MultilabelJaccardIndex(num_labels=3)
            >>> metric.update(randint(2, (20, 3)), randint(2, (20, 3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import MultilabelJaccardIndex
            >>> metric = MultilabelJaccardIndex(num_labels=3)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(randint(2, (20, 3)), randint(2, (20, 3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)