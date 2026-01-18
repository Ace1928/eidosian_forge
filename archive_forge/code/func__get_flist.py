import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform
def _get_flist(root: str, file_path: str, subset: str) -> List[str]:
    f_list = []
    if subset == 'train':
        index = 1
    elif subset == 'dev':
        index = 2
    else:
        index = 3
    with open(file_path, 'r') as f:
        for line in f:
            id, path = line.split()
            if int(id) == index:
                f_list.append(path)
    return sorted(f_list)