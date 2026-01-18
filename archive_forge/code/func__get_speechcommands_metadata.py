import os
from pathlib import Path
from typing import Optional, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
def _get_speechcommands_metadata(filepath: str, path: str) -> Tuple[str, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    reldir, filename = os.path.split(relpath)
    _, label = os.path.split(reldir)
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)
    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)
    return (relpath, SAMPLE_RATE, label, speaker_id, utterance_number)