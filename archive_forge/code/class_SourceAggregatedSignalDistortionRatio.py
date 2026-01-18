from typing import Any, Optional, Sequence, Union
from torch import Tensor, tensor
from torchmetrics.functional.audio.sdr import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class SourceAggregatedSignalDistortionRatio(Metric):
    """`Source-aggregated signal-to-distortion ratio`_ (SA-SDR).

    The SA-SDR is proposed to provide a stable gradient for meeting style source separation, where
    one-speaker and multiple-speaker scenes coexist.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(..., spk, time)``
    - ``target`` (:class:`~torch.Tensor`): float tensor with shape ``(..., spk, time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``sa_sdr`` (:class:`~torch.Tensor`): float scalar tensor with average SA-SDR value over samples

    Args:
        preds: float tensor with shape ``(..., spk, time)``
        target: float tensor with shape ``(..., spk, time)``
        scale_invariant: if True, scale the targets of different speakers with the same alpha
        zero_mean: If to zero mean target and preds or not
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.audio import SourceAggregatedSignalDistortionRatio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn(2, 8000) # [..., spk, time]
        >>> target = torch.randn(2, 8000)
        >>> sasdr = SourceAggregatedSignalDistortionRatio()
        >>> sasdr(preds, target)
        tensor(-41.6579)
        >>> # use with pit
        >>> from torchmetrics.audio import PermutationInvariantTraining
        >>> from torchmetrics.functional.audio import source_aggregated_signal_distortion_ratio
        >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
        >>> target = torch.randn(4, 2, 8000)
        >>> pit = PermutationInvariantTraining(source_aggregated_signal_distortion_ratio,
        ...     mode="permutation-wise", eval_func="max")
        >>> pit(preds, target)
        tensor(-41.2790)

    """
    msum: Tensor
    mnum: Tensor
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    plot_lower_bound: Optional[float] = None
    plot_upper_bound: Optional[float] = None

    def __init__(self, scale_invariant: bool=True, zero_mean: bool=False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(scale_invariant, bool):
            raise ValueError(f'Expected argument `scale_invarint` to be a bool, but got {scale_invariant}')
        self.scale_invariant = scale_invariant
        if not isinstance(zero_mean, bool):
            raise ValueError(f'Expected argument `zero_mean` to be a bool, but got {zero_mean}')
        self.zero_mean = zero_mean
        self.add_state('msum', default=tensor(0.0), dist_reduce_fx='sum')
        self.add_state('mnum', default=tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        mbatch = source_aggregated_signal_distortion_ratio(preds, target, self.scale_invariant, self.zero_mean)
        self.msum += mbatch.sum()
        self.mnum += mbatch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.msum / self.mnum

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
            >>> from torchmetrics.audio import SourceAggregatedSignalDistortionRatio
            >>> metric = SourceAggregatedSignalDistortionRatio()
            >>> metric.update(torch.rand(2,8000), torch.rand(2,8000))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio import SourceAggregatedSignalDistortionRatio
            >>> metric = SourceAggregatedSignalDistortionRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(2,8000), torch.rand(2,8000)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)