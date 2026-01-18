import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def _add_noise_shaping(dithered_waveform: Tensor, waveform: Tensor) -> Tensor:
    """Noise shaping is calculated by error:
    error[n] = dithered[n] - original[n]
    noise_shaped_waveform[n] = dithered[n] + error[n-1]
    """
    wf_shape = waveform.size()
    waveform = waveform.reshape(-1, wf_shape[-1])
    dithered_shape = dithered_waveform.size()
    dithered_waveform = dithered_waveform.reshape(-1, dithered_shape[-1])
    error = dithered_waveform - waveform
    zeros = torch.zeros(1, dtype=error.dtype, device=error.device)
    for index in range(error.size()[0]):
        err = error[index]
        error_offset = torch.cat((zeros, err))
        error[index] = error_offset[:waveform.size()[1]]
    noise_shaped = dithered_waveform + error
    return noise_shaped.reshape(dithered_shape[:-1] + noise_shaped.shape[-1:])