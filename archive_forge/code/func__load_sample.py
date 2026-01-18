import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip
def _load_sample(self, n: int) -> Tuple[torch.Tensor, int, int, str]:
    name = self.names[n]
    wavs = []
    num_frames = None
    for source in self.sources:
        track = self._get_track(name, source)
        wav, sr = torchaudio.load(str(track))
        if sr != _SAMPLE_RATE:
            raise ValueError(f'expected sample rate {_SAMPLE_RATE}, but got {sr}')
        if num_frames is None:
            num_frames = wav.shape[-1]
        elif wav.shape[-1] != num_frames:
            raise ValueError('num_frames do not match across sources')
        wavs.append(wav)
    stacked = torch.stack(wavs)
    return (stacked, _SAMPLE_RATE, num_frames, name)