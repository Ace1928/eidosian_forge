from typing import Any, Callable, Optional, Tuple
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _permutation_invariant_training(preds: Tensor, target: Tensor, metric_func: Callable, mode: Literal['speaker-wise', 'permutation-wise']='speaker-wise', eval_func: Literal['max', 'min']='max', **kwargs: Any) -> Tuple[Tensor, Tensor]:
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> preds = tensor([[[-0.0579,  0.3560, -0.9604], [-0.1719,  0.3205,  0.2951]]])
    >>> target = tensor([[[ 1.0958, -0.1648,  0.5228], [-0.4100,  1.1942, -0.5103]]])
    >>> best_metric, best_perm = _permutation_invariant_training(
    ...     preds, target, _scale_invariant_signal_distortion_ratio)
    >>> best_metric
    tensor([-5.1091])
    >>> best_perm
    tensor([[0, 1]])
    >>> pit_permutate(preds, best_perm)
    tensor([[[-0.0579,  0.3560, -0.9604],
             [-0.1719,  0.3205,  0.2951]]])

    """
    _deprecated_root_import_func('permutation_invariant_training', 'audio')
    return permutation_invariant_training(preds=preds, target=target, metric_func=metric_func, mode=mode, eval_func=eval_func, **kwargs)