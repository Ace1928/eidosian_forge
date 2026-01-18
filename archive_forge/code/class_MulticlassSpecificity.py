from typing import Any, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from torchmetrics.functional.classification.specificity import _specificity_reduce
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class MulticlassSpecificity(MulticlassStatScores):
    """Compute `Specificity`_ for multiclass tasks.

    .. math:: \\text{Specificity} = \\frac{\\text{TN}}{\\text{TN} + \\text{FP}}

    Where :math:`\\text{TN}` and :math:`\\text{FP}` represent the number of true negatives and false positives
    respectively.  The metric is only proper defined when :math:`\\text{TN} + \\text{FP} \\neq 0`. If this case is
    encountered for any class, the metric for that class will be set to 0 and the overall metric may therefore be
    affected in turn.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcs`` (:class:`~torch.Tensor`): The returned shape depends on the ``average`` and ``multidim_average``
      arguments:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the output will be a scalar tensor
          - If ``average=None/'none'``, the shape will be ``(C,)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N,)``
          - If ``average=None/'none'``, the shape will be ``(N, C)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassSpecificity
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassSpecificity(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8889)
        >>> mcs = MulticlassSpecificity(num_classes=3, average=None)
        >>> mcs(preds, target)
        tensor([1.0000, 0.6667, 1.0000])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassSpecificity
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> metric = MulticlassSpecificity(num_classes=3)
        >>> metric(preds, target)
        tensor(0.8889)
        >>> mcs = MulticlassSpecificity(num_classes=3, average=None)
        >>> mcs(preds, target)
        tensor([1.0000, 0.6667, 1.0000])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassSpecificity
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassSpecificity(num_classes=3, multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([0.7500, 0.6556])
        >>> mcs = MulticlassSpecificity(num_classes=3, multidim_average='samplewise', average=None)
        >>> mcs(preds, target)
        tensor([[0.7500, 0.7500, 0.7500],
                [0.8000, 0.6667, 0.5000]])

    """
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Class'

    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, tn, fn = self._final_state()
        return _specificity_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average)

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

            >>> from torch import randint
            >>> # Example plotting a single value per class
            >>> from torchmetrics.classification import MulticlassSpecificity
            >>> metric = MulticlassSpecificity(num_classes=3, average=None)
            >>> metric.update(randint(3, (20,)), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randint
            >>> # Example plotting a multiple values per class
            >>> from torchmetrics.classification import MulticlassSpecificity
            >>> metric = MulticlassSpecificity(num_classes=3, average=None)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randint(3, (20,)), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)