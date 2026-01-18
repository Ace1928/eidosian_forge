from typing import Any, Callable, Optional, Tuple
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _signal_distortion_ratio(preds: Tensor, target: Tensor, use_cg_iter: Optional[int]=None, filter_length: int=512, zero_mean: bool=False, load_diag: Optional[float]=None) -> Tensor:
    """Wrapper for deprecated import.

    >>> import torch
    >>> g = torch.manual_seed(1)
    >>> preds = torch.randn(8000)
    >>> target = torch.randn(8000)
    >>> _signal_distortion_ratio(preds, target)
    tensor(-12.0589)
    >>> # use with permutation_invariant_training
    >>> preds = torch.randn(4, 2, 8000)  # [batch, spk, time]
    >>> target = torch.randn(4, 2, 8000)
    >>> best_metric, best_perm = _permutation_invariant_training(preds, target, _signal_distortion_ratio)
    >>> best_metric
    tensor([-11.6375, -11.4358, -11.7148, -11.6325])
    >>> best_perm
    tensor([[1, 0],
            [0, 1],
            [1, 0],
            [0, 1]])

    """
    _deprecated_root_import_func('signal_distortion_ratio', 'audio')
    return signal_distortion_ratio(preds=preds, target=target, use_cg_iter=use_cg_iter, filter_length=filter_length, zero_mean=zero_mean, load_diag=load_diag)