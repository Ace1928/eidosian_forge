from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics import Metric
from torchmetrics.functional.retrieval.precision_recall_curve import retrieval_precision_recall_curve
from torchmetrics.retrieval.base import _retrieval_aggregate
from torchmetrics.utilities.checks import _check_retrieval_inputs
from torchmetrics.utilities.data import _flexible_bincount, dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_curve
class RetrievalRecallAtFixedPrecision(RetrievalPrecisionRecallCurve):
    """Compute `IR Recall at fixed Precision`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``
    - ``target`` (:class:`~torch.Tensor`): A long or bool tensor of shape ``(N, ...)``
    - ``indexes`` (:class:`~torch.Tensor`): A long tensor of shape ``(N, ...)`` which indicate to which query a
      prediction belongs

    .. note:: All ``indexes``, ``preds`` and ``target`` must have the same dimension.

    .. note::
        Predictions will be first grouped by ``indexes`` and then `RetrievalRecallAtFixedPrecision`
        will be computed as the mean of the `RetrievalRecallAtFixedPrecision` over each query.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``max_recall`` (:class:`~torch.Tensor`): A tensor with the maximum recall value
      retrieved documents.
    - ``best_k`` (:class:`~torch.Tensor`): A tensor with the best k corresponding to the maximum recall value

    Args:
        min_precision: float value specifying minimum precision threshold.
        max_k: Calculate recall and precision for all possible top k from 1 to max_k
               (default: `None`, which considers all possible top k)
        adaptive_k: adjust `k` to `min(k, number of documents)` for each query
        empty_target_action:
            Specify what to do with queries that do not have at least a positive ``target``. Choose from:

            - ``'neg'``: those queries count as ``0.0`` (default)
            - ``'pos'``: those queries count as ``1.0``
            - ``'skip'``: skip those queries; if all queries are skipped, ``0.0`` is returned
            - ``'error'``: raise a ``ValueError``

        ignore_index:
            Ignore predictions where the target is equal to this number.
        kwargs:
            Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``empty_target_action`` is not one of ``error``, ``skip``, ``neg`` or ``pos``.
        ValueError:
            If ``ignore_index`` is not `None` or an integer.
        ValueError:
            If ``min_precision`` parameter is not float or between 0 and 1.
        ValueError:
            If ``max_k`` parameter is not `None` or an integer larger than 0.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.retrieval import RetrievalRecallAtFixedPrecision
        >>> indexes = tensor([0, 0, 0, 0, 1, 1, 1])
        >>> preds = tensor([0.4, 0.01, 0.5, 0.6, 0.2, 0.3, 0.5])
        >>> target = tensor([True, False, False, True, True, False, True])
        >>> r = RetrievalRecallAtFixedPrecision(min_precision=0.8)
        >>> r(preds, target, indexes=indexes)
        (tensor(0.5000), tensor(1))

    """
    higher_is_better = True

    def __init__(self, min_precision: float=0.0, max_k: Optional[int]=None, adaptive_k: bool=False, empty_target_action: str='neg', ignore_index: Optional[int]=None, **kwargs: Any) -> None:
        super().__init__(max_k=max_k, adaptive_k=adaptive_k, empty_target_action=empty_target_action, ignore_index=ignore_index, **kwargs)
        if not (isinstance(min_precision, float) and 0.0 <= min_precision <= 1.0):
            raise ValueError('`min_precision` has to be a positive float between 0 and 1')
        self.min_precision = min_precision

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Compute metric."""
        precisions, recalls, top_k = super().compute()
        return _retrieval_recall_at_fixed_precision(precisions, recalls, top_k, self.min_precision)

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

            >>> import torch
            >>> from torchmetrics.retrieval import RetrievalRecallAtFixedPrecision
            >>> # Example plotting a single value
            >>> metric = RetrievalRecallAtFixedPrecision(min_precision=0.5)
            >>> metric.update(torch.rand(10,), torch.randint(2, (10,)), indexes=torch.randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> import torch
            >>> from torchmetrics.retrieval import RetrievalRecallAtFixedPrecision
            >>> # Example plotting multiple values
            >>> metric = RetrievalRecallAtFixedPrecision(min_precision=0.5)
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(10,), torch.randint(2, (10,)), indexes=torch.randint(2,(10,)))[0])
            >>> fig, ax = metric.plot(values)

        """
        val = val or self.compute()[0]
        return self._plot(val, ax)