import os
from pathlib import Path
from typing import Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
class LIBRITTS(Dataset):
    """*LibriTTS* :cite:`Zen2019LibriTTSAC` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriTTS"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """
    _ext_original_txt = '.original.txt'
    _ext_normalized_txt = '.normalized.txt'
    _ext_audio = '.wav'

    def __init__(self, root: Union[str, Path], url: str=URL, folder_in_archive: str=FOLDER_IN_ARCHIVE, download: bool=False) -> None:
        if url in ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']:
            ext_archive = '.tar.gz'
            base_url = 'http://www.openslr.org/resources/60/'
            url = os.path.join(base_url, url + ext_archive)
        root = os.fspath(root)
        basename = os.path.basename(url)
        archive = os.path.join(root, basename)
        basename = basename.split('.')[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)
        self._path = os.path.join(root, folder_in_archive)
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                _extract_tar(archive)
        elif not os.path.exists(self._path):
            raise RuntimeError(f"The path {self._path} doesn't exist. Please check the ``root`` path or set `download=True` to download it")
        self._walker = sorted((str(p.stem) for p in Path(self._path).glob('*/*/*' + self._ext_audio)))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int, int, str]:
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
                Original text
            str:
                Normalized text
            int:
                Speaker ID
            int:
                Chapter ID
            str:
                Utterance ID
        """
        fileid = self._walker[n]
        return load_libritts_item(fileid, self._path, self._ext_audio, self._ext_original_txt, self._ext_normalized_txt)

    def __len__(self) -> int:
        return len(self._walker)