import os
from pathlib import Path
from typing import Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
class LIBRISPEECH(Dataset):
    """*LibriSpeech* :cite:`7178964` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """
    _ext_txt = '.trans.txt'
    _ext_audio = '.flac'

    def __init__(self, root: Union[str, Path], url: str=URL, folder_in_archive: str=FOLDER_IN_ARCHIVE, download: bool=False) -> None:
        self._url = url
        if url not in _DATA_SUBSETS:
            raise ValueError(f"Invalid url '{url}' given; please provide one of {_DATA_SUBSETS}.")
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)
        self._path = os.path.join(root, folder_in_archive, url)
        if not os.path.isdir(self._path):
            if download:
                _download_librispeech(root, url)
            else:
                raise RuntimeError(f'Dataset not found at {self._path}. Please set `download=True` to download the dataset.')
        self._walker = sorted((str(p.stem) for p in Path(self._path).glob('*/*/*' + self._ext_audio)))

    def get_metadata(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        fileid = self._walker[n]
        return _get_librispeech_metadata(fileid, self._archive, self._url, self._ext_audio, self._ext_txt)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Chapter ID
            int:
                Utterance ID
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self._walker)