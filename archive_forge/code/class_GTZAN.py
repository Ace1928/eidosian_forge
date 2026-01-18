import os
from pathlib import Path
from typing import Optional, Tuple, Union
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar
class GTZAN(Dataset):
    """*GTZAN* :cite:`tzanetakis_essl_cook_2001` dataset.

    Note:
        Please see http://marsyas.info/downloads/datasets.html if you are planning to use
        this dataset to publish results.

    Note:
        As of October 2022, the download link is not currently working. Setting ``download=True``
        in GTZAN dataset will result in a URL connection error.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"http://opihi.cs.uvic.ca/sound/genres.tar.gz"``)
        folder_in_archive (str, optional): The top-level directory of the dataset.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """
    _ext_audio = '.wav'

    def __init__(self, root: Union[str, Path], url: str=URL, folder_in_archive: str=FOLDER_IN_ARCHIVE, download: bool=False, subset: Optional[str]=None) -> None:
        root = os.fspath(root)
        self.root = root
        self.url = url
        self.folder_in_archive = folder_in_archive
        self.download = download
        self.subset = subset
        if subset is not None and subset not in ['training', 'validation', 'testing']:
            raise ValueError("When `subset` is not None, it must be one of ['training', 'validation', 'testing'].")
        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                _extract_tar(archive)
        if not os.path.isdir(self._path):
            raise RuntimeError('Dataset not found. Please use `download=True` to download it.')
        if self.subset is None:
            self._walker = []
            root = os.path.expanduser(self._path)
            for directory in gtzan_genres:
                fulldir = os.path.join(root, directory)
                if not os.path.exists(fulldir):
                    continue
                songs_in_genre = os.listdir(fulldir)
                songs_in_genre.sort()
                for fname in songs_in_genre:
                    name, ext = os.path.splitext(fname)
                    if ext.lower() == '.wav' and '.' in name:
                        genre, num = name.split('.')
                        if genre in gtzan_genres and len(num) == 5 and num.isdigit():
                            self._walker.append(name)
        elif self.subset == 'training':
            self._walker = filtered_train
        elif self.subset == 'validation':
            self._walker = filtered_valid
        elif self.subset == 'testing':
            self._walker = filtered_test

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
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
                Label
        """
        fileid = self._walker[n]
        item = load_gtzan_item(fileid, self._path, self._ext_audio)
        waveform, sample_rate, label = item
        return (waveform, sample_rate, label)

    def __len__(self) -> int:
        return len(self._walker)