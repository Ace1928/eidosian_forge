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
class MultilabelAveragePrecision(MultilabelPrecisionRecallCurve):
    """Compute the average precision (AP) score for multilabel tasks.

    The AP score summarizes a precision-recall curve as an weighted mean of precisions at each threshold, with the
    difference in recall from the previous threshold as weight:

    .. math::
        AP = \\sum_{n} (R_n - R_{n-1}) P_n

    where :math:`P_n, R_n` is the respective precision and recall at threshold index :math:`n`. This value is
    equivalent to the area under the precision-recall curve (AUPRC).

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)`` containing probabilities or logits
      for each observation. If preds has values outside [0,1] range we consider the input to be logits and will auto
      apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)`` containing ground truth labels, and
      therefore only contain {0,1} values (except if `ignore_index` is specified).

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mlap`` (:class:`~torch.Tensor`): If `average=None|"none"` then a 1d tensor of shape (n_classes, ) will be
      returned with AP score per class. If `average="micro|macro"|"weighted"` then a single scalar is returned.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned
    version that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate
    the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the
    `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\\mathcal{O}(n_{thresholds} \\times n_{labels})` (constant memory).

    Args:
        num_labels: Integer specifying the number of labels
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum score over all labels
            - ``macro``: Calculate score for each label and average them
            - ``weighted``: calculates score for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates score for each label and applies no reduction
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification import MultilabelAveragePrecision
        >>> preds = tensor([[0.75, 0.05, 0.35],
        ...                 [0.45, 0.75, 0.05],
        ...                 [0.05, 0.55, 0.75],
        ...                 [0.05, 0.65, 0.05]])
        >>> target = tensor([[1, 0, 1],
        ...                  [0, 0, 0],
        ...                  [0, 1, 1],
        ...                  [1, 1, 1]])
        >>> metric = MultilabelAveragePrecision(num_labels=3, average="macro", thresholds=None)
        >>> metric(preds, target)
        tensor(0.7500)
        >>> mlap = MultilabelAveragePrecision(num_labels=3, average=None, thresholds=None)
        >>> mlap(preds, target)
        tensor([0.7500, 0.5833, 0.9167])
        >>> mlap = MultilabelAveragePrecision(num_labels=3, average="macro", thresholds=5)
        >>> mlap(preds, target)
        tensor(0.7778)
        >>> mlap = MultilabelAveragePrecision(num_labels=3, average=None, thresholds=5)
        >>> mlap(preds, target)
        tensor([0.7500, 0.6667, 0.9167])

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Label'

    def __init__(self, num_labels: int, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(num_labels=num_labels, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs)
        if validate_args:
            _multilabel_average_precision_arg_validation(num_labels, average, thresholds, ignore_index)
        self.average = average
        self.validate_args = validate_args

    def compute(self) -> Tensor:
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _multilabel_average_precision_compute(state, self.num_labels, self.average, self.thresholds, self.ignore_index)

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

            >>> # Example plotting a single
            >>> import torch
            >>> from torchmetrics.classification import MultilabelAveragePrecision
            >>> metric = MultilabelAveragePrecision(num_labels=3)
            >>> metric.update(torch.rand(20,3), torch.randint(2, (20,3)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.classification import MultilabelAveragePrecision
            >>> metric = MultilabelAveragePrecision(num_labels=3)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(20,3), torch.randint(2, (20,3))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)