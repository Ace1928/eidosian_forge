import numpy as np
import torch
from contextlib import contextmanager
from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_PPC, IS_MACOS, IS_WINDOWS
Calculates the signal to noise ratio and returns the signal and noise
    power, as well as the SNR in dB.
    If the input is a list/tuple this function is called recursively on each
    element. The result will have the same nested structure as the inputs.

    Args:
        x, x_hat: Either a tensor or a nested list/tuple of tensors.
    Returns:
        signal, noise, SNR(in dB): Either floats or a nested list of floats
    