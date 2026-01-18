from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.running import Running
class CatMetric(BaseAggregator):
    """Concatenate a stream of values.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with concatenated values over all input received

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
        >>> from torchmetrics.aggregation import CatMetric
        >>> metric = CatMetric()
        >>> metric.update(1)
        >>> metric.update(tensor([2, 3]))
        >>> metric.compute()
        tensor([1., 2., 3.])

    """
    value: Tensor

    def __init__(self, nan_strategy: Union[str, float]='warn', **kwargs: Any) -> None:
        super().__init__('cat', [], nan_strategy, **kwargs)

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():
            self.value.append(value)

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        if isinstance(self.value, list) and self.value:
            return dim_zero_cat(self.value)
        return self.value