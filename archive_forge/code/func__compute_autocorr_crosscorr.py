import math
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.linalg import norm
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.imports import _FAST_BSS_EVAL_AVAILABLE
def _compute_autocorr_crosscorr(target: Tensor, preds: Tensor, corr_len: int) -> Tuple[Tensor, Tensor]:
    """Compute the auto correlation of `target` and the cross correlation of `target` and `preds`.

    This calculation is done using the fast Fourier transform (FFT). Let's denotes the symmetric Toeplitz metric of the
    auto correlation of `target` as `R`, the cross correlation as 'b', then solving the equation `Rh=b` could have `h`
    as the coordinate of `preds` in the column space of the `corr_len` shifts of `target`.

    Args:
        target: the target (reference) signal of shape [..., time]
        preds: the preds (estimated) signal of shape [..., time]
        corr_len: the length of the auto correlation and cross correlation

    Returns:
        the auto correlation of `target` of shape [..., corr_len]
        the cross correlation of `target` and `preds` of shape [..., corr_len]

    """
    n_fft = 2 ** math.ceil(math.log2(preds.shape[-1] + target.shape[-1] - 1))
    t_fft = torch.fft.rfft(target, n=n_fft, dim=-1)
    r_0 = torch.fft.irfft(t_fft.real ** 2 + t_fft.imag ** 2, n=n_fft)[..., :corr_len]
    p_fft = torch.fft.rfft(preds, n=n_fft, dim=-1)
    b = torch.fft.irfft(t_fft.conj() * p_fft, n=n_fft, dim=-1)[..., :corr_len]
    return (r_0, b)