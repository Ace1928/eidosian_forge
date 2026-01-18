import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform
class VoxCeleb1Identification(VoxCeleb1):
    """*VoxCeleb1* :cite:`nagrani2017voxceleb` dataset for speaker identification task.

    Each data sample contains the waveform, sample rate, speaker id, and the file id.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset (str, optional): Subset of the dataset to use. Options: ["train", "dev", "test"]. (Default: ``"train"``)
        meta_url (str, optional): The url of meta file that contains the list of subset labels and file paths.
            The format of each row is ``subset file_path". For example: ``1 id10006/nLEBBc9oIFs/00003.wav``.
            ``1``, ``2``, ``3`` mean ``train``, ``dev``, and ``test`` subest, respectively.
            (Default: ``"https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (Default: ``False``).

    Note:
        The file structure of `VoxCeleb1Identification` dataset is as follows:

        └─ root/

         └─ wav/

         └─ speaker_id folders

        Users who pre-downloaded the ``"vox1_dev_wav.zip"`` and ``"vox1_test_wav.zip"`` files need to move
        the extracted files into the same ``root`` directory.
    """

    def __init__(self, root: Union[str, Path], subset: str='train', meta_url: str=_IDEN_SPLIT_URL, download: bool=False) -> None:
        super().__init__(root, download)
        if subset not in ['train', 'dev', 'test']:
            raise ValueError("`subset` must be one of ['train', 'dev', 'test']")
        meta_list_path = os.path.join(root, os.path.basename(meta_url))
        if not os.path.exists(meta_list_path):
            download_url_to_file(meta_url, meta_list_path)
        self._flist = _get_flist(self._path, meta_list_path, subset)

    def get_metadata(self, n: int) -> Tuple[str, int, int, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Sample rate
            int:
                Speaker ID
            str:
                File ID
        """
        file_path = self._flist[n]
        file_id = _get_file_id(file_path, self._ext_audio)
        speaker_id = file_id.split('-')[0]
        speaker_id = int(speaker_id[3:])
        return (file_path, SAMPLE_RATE, speaker_id, file_id)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            int:
                Speaker ID
            str:
                File ID
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self._flist)