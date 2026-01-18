import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip
def _collect_songs(self):
    if self.split == 'validation':
        return _VALIDATION_SET
    path = Path(self._path)
    names = []
    for root, folders, _ in os.walk(path, followlinks=True):
        root = Path(root)
        if root.name.startswith('.') or folders or root == path:
            continue
        name = str(root.relative_to(path))
        if self.split and name in _VALIDATION_SET:
            continue
        names.append(name)
    return sorted(names)