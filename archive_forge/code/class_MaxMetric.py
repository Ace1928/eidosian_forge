from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.running import Running
class MaxMetric(BaseAggregator):
    """Aggregate a stream of value into their maximum value.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated maximum value over all inputs received

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
        >>> from torch import tensor
        >>> from torchmetrics.aggregation import MaxMetric
        >>> metric = MaxMetric()
        >>> metric.update(1)
        >>> metric.update(tensor([2, 3]))
        >>> metric.compute()
        tensor(3.)

    """
    full_state_update: bool = True
    max_value: Tensor

    def __init__(self, nan_strategy: Union[str, float]='warn', **kwargs: Any) -> None:
        super().__init__('max', -torch.tensor(float('inf'), dtype=torch.get_default_dtype()), nan_strategy, state_name='max_value', **kwargs)

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():
            self.max_value = torch.max(self.max_value, torch.max(value))

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
            >>> from torchmetrics.aggregation import MaxMetric
            >>> metric = MaxMetric()
            >>> metric.update([1, 2, 3])
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.aggregation import MaxMetric
            >>> metric = MaxMetric()
            >>> values = [ ]
            >>> for i in range(10):
            ...     values.append(metric(i))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)