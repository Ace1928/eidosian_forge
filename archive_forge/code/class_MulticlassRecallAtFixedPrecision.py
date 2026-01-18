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
class MulticlassRecallAtFixedPrecision(MulticlassPrecisionRecallCurve):
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for
    a given precision level.

    For multiclass the metric is calculated by iteratively treating each class as the positive class and all other
    classes as the negative, which is referred to as the one-vs-rest approach. One-vs-one is currently not supported by
    this metric.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply softmax per sample.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain values in the [0, n_classes-1] range (except if `ignore_index`
      is specified).

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns a tuple of either 2 tensors or 2 lists containing:

    - ``recall`` (:class:`~torch.Tensor`): A 1d tensor of size ``(n_classes, )`` with the maximum recall for the
      given precision level per class
    - ``threshold`` (:class:`~torch.Tensor`): A 1d tensor of size ``(n_classes, )`` with the corresponding threshold
      level per class

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to ``None`` will activate the
       non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting the `thresholds`
       argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\\mathcal{O}(n_{thresholds} \\times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifying the number of classes
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an ``int`` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an ``list`` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d :class:`~torch.Tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassRecallAtFixedPrecision
        >>> preds = tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                 [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                 [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                 [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = tensor([0, 1, 3, 2])
        >>> metric = MulticlassRecallAtFixedPrecision(num_classes=5, min_precision=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor([1., 1., 0., 0., 0.]), tensor([7.5000e-01, 7.5000e-01, 1.0000e+06, 1.0000e+06, 1.0000e+06]))
        >>> mcrafp = MulticlassRecallAtFixedPrecision(num_classes=5, min_precision=0.5, thresholds=5)
        >>> mcrafp(preds, target)
        (tensor([1., 1., 0., 0., 0.]), tensor([7.5000e-01, 7.5000e-01, 1.0000e+06, 1.0000e+06, 1.0000e+06]))

    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Class'

    def __init__(self, num_classes: int, min_precision: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(num_classes=num_classes, thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs)
        if validate_args:
            _multiclass_recall_at_fixed_precision_arg_validation(num_classes, min_precision, thresholds, ignore_index)
        self.validate_args = validate_args
        self.min_precision = min_precision

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _multiclass_recall_at_fixed_precision_arg_compute(state, self.num_classes, self.thresholds, self.min_precision)

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

            >>> from torch import rand, randint
            >>> # Example plotting a single value per class
            >>> from torchmetrics.classification import MulticlassRecallAtFixedPrecision
            >>> metric = MulticlassRecallAtFixedPrecision(num_classes=3, min_precision=0.5)
            >>> metric.update(rand(20, 3).softmax(dim=-1), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()  # the returned plot only shows the maximum recall value by default

        .. plot::
            :scale: 75

            >>> from torch import rand, randint
            >>> # Example plotting a multiple values per class
            >>> from torchmetrics.classification import MulticlassRecallAtFixedPrecision
            >>> metric = MulticlassRecallAtFixedPrecision(num_classes=3, min_precision=0.5)
            >>> values = []
            >>> for _ in range(20):
            ...     # we index by 0 such that only the maximum recall value is plotted
            ...     values.append(metric(rand(20, 3).softmax(dim=-1), randint(3, (20,)))[0])
            >>> fig_, ax_ = metric.plot(values)

        """
        val = val or self.compute()[0]
        return self._plot(val, ax)