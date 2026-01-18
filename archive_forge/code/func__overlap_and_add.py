import warnings
from typing import List, Optional, Union
import torch
from torchaudio.functional import fftconvolve
def _overlap_and_add(waveform, stride):
    num_frames, frame_size = waveform.shape[-2:]
    numel = (num_frames - 1) * stride + frame_size
    buffer = torch.zeros(waveform.shape[:-2] + (numel,), device=waveform.device, dtype=waveform.dtype)
    for i in range(num_frames):
        start = i * stride
        end = start + frame_size
        buffer[..., start:end] += waveform[..., i, :]
    return buffer