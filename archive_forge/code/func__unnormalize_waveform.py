import logging
import os
import torch
from torchaudio._internal import download_url_to_file, module_utils as _mod_utils
def _unnormalize_waveform(waveform: torch.Tensor, bits: int) -> torch.Tensor:
    """Transform waveform [-1, 1] to label [0, 2 ** bits - 1]"""
    waveform = torch.clamp(waveform, -1, 1)
    waveform = (waveform + 1.0) * (2 ** bits - 1) / 2
    return torch.clamp(waveform, 0, 2 ** bits - 1).int()