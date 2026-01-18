import os
from pathlib import Path
from typing import List, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
def _load_item(self, fileid: str, path: str):
    labels = [int(c) for c in fileid.split('_')]
    file_audio = os.path.join(path, fileid + '.wav')
    waveform, sample_rate = torchaudio.load(file_audio)
    return (waveform, sample_rate, labels)