from typing import Any, Callable, Optional
from typing_extensions import Literal
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.utilities.prints import _deprecated_root_import_class
class _SignalNoiseRatio(SignalNoiseRatio):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> target = tensor([3.0, -0.5, 2.0, 7.0])
    >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
    >>> snr = _SignalNoiseRatio()
    >>> snr(preds, target)
    tensor(16.1805)

    """

    def __init__(self, zero_mean: bool=False, **kwargs: Any) -> None:
        _deprecated_root_import_class('SignalNoiseRatio', 'audio')
        super().__init__(zero_mean=zero_mean, **kwargs)