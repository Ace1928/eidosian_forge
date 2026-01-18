import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform
class VoxCeleb1Verification(VoxCeleb1):
    """*VoxCeleb1* :cite:`nagrani2017voxceleb` dataset for speaker verification task.

    Each data sample contains a pair of waveforms, sample rate, the label indicating if they are
    from the same speaker, and the file ids.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        meta_url (str, optional): The url of meta file that contains a list of utterance pairs
            and the corresponding labels. The format of each row is ``label file_path1 file_path2".
            For example: ``1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav``.
            ``1`` means the two utterances are from the same speaker, ``0`` means not.
            (Default: ``"https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (Default: ``False``).

    Note:
        The file structure of `VoxCeleb1Verification` dataset is as follows:

        └─ root/

         └─ wav/

         └─ speaker_id folders

        Users who pre-downloaded the ``"vox1_dev_wav.zip"`` and ``"vox1_test_wav.zip"`` files need to move
        the extracted files into the same ``root`` directory.
    """

    def __init__(self, root: Union[str, Path], meta_url: str=_VERI_TEST_URL, download: bool=False) -> None:
        super().__init__(root, download)
        meta_list_path = os.path.join(root, os.path.basename(meta_url))
        if not os.path.exists(meta_list_path):
            download_url_to_file(meta_url, meta_list_path)
        self._flist = _get_paired_flist(self._path, meta_list_path)

    def get_metadata(self, n: int) -> Tuple[str, str, int, int, str, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepaths instead of waveforms,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample

        Returns:
            Tuple of the following items;

            str:
                Path to audio file of speaker 1
            str:
                Path to audio file of speaker 2
            int:
                Sample rate
            int:
                Label
            str:
                File ID of speaker 1
            str:
                File ID of speaker 2
        """
        label, file_path_spk1, file_path_spk2 = self._flist[n]
        label = int(label)
        file_id_spk1 = _get_file_id(file_path_spk1, self._ext_audio)
        file_id_spk2 = _get_file_id(file_path_spk2, self._ext_audio)
        return (file_path_spk1, file_path_spk2, SAMPLE_RATE, label, file_id_spk1, file_id_spk2)

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, int, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform of speaker 1
            Tensor:
                Waveform of speaker 2
            int:
                Sample rate
            int:
                Label
            str:
                File ID of speaker 1
            str:
                File ID of speaker 2
        """
        metadata = self.get_metadata(n)
        waveform_spk1 = _load_waveform(self._path, metadata[0], metadata[2])
        waveform_spk2 = _load_waveform(self._path, metadata[1], metadata[2])
        return (waveform_spk1, waveform_spk2) + metadata[2:]

    def __len__(self) -> int:
        return len(self._flist)