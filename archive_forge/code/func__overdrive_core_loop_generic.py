import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def _overdrive_core_loop_generic(waveform: Tensor, temp: Tensor, last_in: Tensor, last_out: Tensor, output_waveform: Tensor):
    for i in range(waveform.shape[-1]):
        last_out = temp[:, i] - last_in + 0.995 * last_out
        last_in = temp[:, i]
        output_waveform[:, i] = waveform[:, i] * 0.5 + last_out * 0.75