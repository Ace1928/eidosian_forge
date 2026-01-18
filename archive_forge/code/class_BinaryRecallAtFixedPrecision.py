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
class BinaryRecallAtFixedPrecision(BinaryPrecisionRecallCurve):
    """Compute the highest possible recall value given the minimum precision thresholds provided.

    This is done by first calculating the precision-recall curve for different thresholds and the find the recall for
    a given precision level.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input
      to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified). The value
      1 always encodes the positive class.

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``recall`` (:class:`~torch.Tensor`): A scalar tensor with the maximum recall for the given precision level
    - ``threshold`` (:class:`~torch.Tensor`): A scalar tensor with the corresponding threshold level

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a
       binned version that is less accurate but more memory efficient. Setting the `thresholds` argument to ``None``
       will activate the non-binned  version that uses memory of size :math:`\\mathcal{O}(n_{samples})` whereas setting
       the `thresholds` argument to either an integer, list or a 1d tensor will use a binned version that uses memory
       of size :math:`\\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        min_precision: float value specifying minimum precision threshold.
        thresholds:
            Can be one of:

            - If set to ``None``, will use a non-binned approach where thresholds are dynamically calculated from
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
        >>> from torchmetrics.classification import BinaryRecallAtFixedPrecision
        >>> preds = tensor([0, 0.5, 0.7, 0.8])
        >>> target = tensor([0, 1, 1, 0])
        >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5, thresholds=None)
        >>> metric(preds, target)
        (tensor(1.), tensor(0.5000))
        >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5, thresholds=5)
        >>> metric(preds, target)
        (tensor(1.), tensor(0.5000))

    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    def __init__(self, min_precision: float, thresholds: Optional[Union[int, List[float], Tensor]]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(thresholds, ignore_index, validate_args=False, **kwargs)
        if validate_args:
            _binary_recall_at_fixed_precision_arg_validation(min_precision, thresholds, ignore_index)
        self.validate_args = validate_args
        self.min_precision = min_precision

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target)) if self.thresholds is None else self.confmat
        return _binary_recall_at_fixed_precision_compute(state, self.thresholds, self.min_precision)

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
            >>> # Example plotting a single value
            >>> from torchmetrics.classification import BinaryRecallAtFixedPrecision
            >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5)
            >>> metric.update(rand(10), randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()  # the returned plot only shows the maximum recall value by default

        .. plot::
            :scale: 75

            >>> from torch import rand, randint
            >>> # Example plotting multiple values
            >>> from torchmetrics.classification import BinaryRecallAtFixedPrecision
            >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     # we index by 0 such that only the maximum recall value is plotted
            ...     values.append(metric(rand(10), randint(2,(10,)))[0])
            >>> fig_, ax_ = metric.plot(values)

        """
        val = val or self.compute()[0]
        return self._plot(val, ax)