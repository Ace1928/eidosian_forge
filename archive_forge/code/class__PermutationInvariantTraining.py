from typing import Any, Callable, Optional
from typing_extensions import Literal
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.utilities.prints import _deprecated_root_import_class
class _PermutationInvariantTraining(PermutationInvariantTraining):
    """Wrapper for deprecated import.

    >>> import torch
    >>> from torchmetrics.functional import scale_invariant_signal_noise_ratio
    >>> _ = torch.manual_seed(42)
    >>> preds = torch.randn(3, 2, 5) # [batch, spk, time]
    >>> target = torch.randn(3, 2, 5) # [batch, spk, time]
    >>> pit = _PermutationInvariantTraining(scale_invariant_signal_noise_ratio,
    ...     mode="speaker-wise", eval_func="max")
    >>> pit(preds, target)
    tensor(-2.1065)

    """

    def __init__(self, metric_func: Callable, mode: Literal['speaker-wise', 'permutation-wise']='speaker-wise', eval_func: Literal['max', 'min']='max', **kwargs: Any) -> None:
        _deprecated_root_import_class('PermutationInvariantTraining', 'audio')
        super().__init__(metric_func=metric_func, mode=mode, eval_func=eval_func, **kwargs)