from typing import Any, Callable, Optional
from typing_extensions import Literal
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics.utilities.prints import _deprecated_root_import_class
class _SignalDistortionRatio(SignalDistortionRatio):
    """Wrapper for deprecated import.

    >>> import torch
    >>> g = torch.manual_seed(1)
    >>> preds = torch.randn(8000)
    >>> target = torch.randn(8000)
    >>> sdr = _SignalDistortionRatio()
    >>> sdr(preds, target)
    tensor(-12.0589)
    >>> # use with pit
    >>> from torchmetrics.functional import signal_distortion_ratio
    >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
    >>> target = torch.randn(4, 2, 8000)
    >>> pit = _PermutationInvariantTraining(signal_distortion_ratio,
    ...     mode="speaker-wise", eval_func="max")
    >>> pit(preds, target)
    tensor(-11.6051)

    """

    def __init__(self, use_cg_iter: Optional[int]=None, filter_length: int=512, zero_mean: bool=False, load_diag: Optional[float]=None, **kwargs: Any) -> None:
        _deprecated_root_import_class('SignalDistortionRatio', 'audio')
        super().__init__(use_cg_iter=use_cg_iter, filter_length=filter_length, zero_mean=zero_mean, load_diag=load_diag, **kwargs)