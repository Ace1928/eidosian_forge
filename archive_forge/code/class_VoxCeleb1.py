import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform
class VoxCeleb1(Dataset):
    """*VoxCeleb1* :cite:`nagrani2017voxceleb` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (Default: ``False``).
    """
    _ext_audio = '.wav'

    def __init__(self, root: Union[str, Path], download: bool=False) -> None:
        root = os.fspath(root)
        self._path = os.path.join(root, 'wav')
        if not os.path.isdir(self._path):
            if not download:
                raise RuntimeError(f'Dataset not found at {self._path}. Please set `download=True` to download the dataset.')
            _download_extract_wavs(root)

    def get_metadata(self, n: int):
        raise NotImplementedError

    def __getitem__(self, n: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError