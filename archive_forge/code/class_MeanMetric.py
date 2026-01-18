from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.running import Running
class MeanMetric(BaseAggregator):
    """Aggregate a stream of value into their mean value.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.
    - ``weight`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float value with
      arbitrary shape ``(...,)``. Needs to be broadcastable with the shape of ``value`` tensor.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated (weighted) mean over all inputs received

    Args:
       nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torchmetrics.aggregation import MeanMetric
        >>> metric = MeanMetric()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(2.)

    """
    mean_value: Tensor

    def __init__(self, nan_strategy: Union[str, float]='warn', **kwargs: Any) -> None:
        super().__init__('sum', torch.tensor(0.0, dtype=torch.get_default_dtype()), nan_strategy, state_name='mean_value', **kwargs)
        self.add_state('weight', default=torch.tensor(0.0, dtype=torch.get_default_dtype()), dist_reduce_fx='sum')

    def update(self, value: Union[float, Tensor], weight: Union[float, Tensor]=1.0) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
            weight: Either a float or tensor containing weights for calculating
                the average. Shape of weight should be able to broadcast with
                the shape of `value`. Default to `1.0` corresponding to simple
                harmonic average.

        """
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is not None and (not isinstance(weight, Tensor)):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)
        if value.numel() == 0:
            return
        self.mean_value += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.mean_value / self.weight

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
            >>> from torchmetrics.aggregation import MeanMetric
            >>> metric = MeanMetric()
            >>> metric.update([1, 2, 3])
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.aggregation import MeanMetric
            >>> metric = MeanMetric()
            >>> values = [ ]
            >>> for i in range(10):
            ...     values.append(metric([i, i+1]))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)