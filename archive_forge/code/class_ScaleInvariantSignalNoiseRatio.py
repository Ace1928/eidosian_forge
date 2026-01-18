from typing import Any, Optional, Sequence, Union
from torch import Tensor, tensor
from torchmetrics.functional.audio.snr import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class ScaleInvariantSignalNoiseRatio(Metric):
    """Calculate `Scale-invariant signal-to-noise ratio`_ (SI-SNR) metric for evaluating quality of audio.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``si_snr`` (:class:`~torch.Tensor`): float scalar tensor with average SI-SNR value over samples

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            if target and preds have a different shape

    Example:
        >>> import torch
        >>> from torch import tensor
        >>> from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
        >>> target = tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_snr = ScaleInvariantSignalNoiseRatio()
        >>> si_snr(preds, target)
        tensor(15.0918)

    """
    is_differentiable = True
    sum_si_snr: Tensor
    total: Tensor
    higher_is_better = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state('sum_si_snr', default=tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        si_snr_batch = scale_invariant_signal_noise_ratio(preds=preds, target=target)
        self.sum_si_snr += si_snr_batch.sum()
        self.total += si_snr_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_si_snr / self.total

    def plot(self, val: Union[Tensor, Sequence[Tensor], None]=None, ax: Optional[_AX_TYPE]=None) -> _PLOT_OUT_TYPE:
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
            >>> import torch
            >>> from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
            >>> metric = ScaleInvariantSignalNoiseRatio()
            >>> metric.update(torch.rand(4), torch.rand(4))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
            >>> metric = ScaleInvariantSignalNoiseRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(4), torch.rand(4)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)