import numpy as np
import torch
from contextlib import contextmanager
from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_PPC, IS_MACOS, IS_WINDOWS
def _snr(x, x_hat):
    """Calculates the signal to noise ratio and returns the signal and noise
    power, as well as the SNR in dB.
    If the input is a list/tuple this function is called recursively on each
    element. The result will have the same nested structure as the inputs.

    Args:
        x, x_hat: Either a tensor or a nested list/tuple of tensors.
    Returns:
        signal, noise, SNR(in dB): Either floats or a nested list of floats
    """
    if isinstance(x, (list, tuple)):
        assert len(x) == len(x_hat)
        res = []
        for idx in range(len(x)):
            res.append(_snr(x[idx], x_hat[idx]))
        return res
    if x_hat.is_quantized:
        x_hat = x_hat.dequantize()
    if x.is_quantized:
        x = x.dequantize()
    noise = (x - x_hat).norm()
    if noise == 0:
        return (0.0, float('inf'), float('inf'))
    signal = x.norm()
    snr = signal / noise
    snr_db = 20 * snr.log10()
    return (signal, noise, snr_db)