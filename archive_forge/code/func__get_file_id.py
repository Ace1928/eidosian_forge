import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform
def _get_file_id(file_path: str, _ext_audio: str):
    speaker_id, youtube_id, utterance_id = file_path.split('/')[-3:]
    utterance_id = utterance_id.replace(_ext_audio, '')
    file_id = '-'.join([speaker_id, youtube_id, utterance_id])
    return file_id