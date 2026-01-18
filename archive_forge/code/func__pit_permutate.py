from typing import Any, Callable, Optional, Tuple
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _pit_permutate(preds: Tensor, perm: Tensor) -> Tensor:
    """Wrapper for deprecated import."""
    _deprecated_root_import_func('pit_permutate', 'audio')
    return pit_permutate(preds=preds, perm=perm)