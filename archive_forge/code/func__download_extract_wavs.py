import os
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform
def _download_extract_wavs(root: str):
    for archive in ['dev', 'test']:
        archive_name = _ARCHIVE_CONFIGS[archive]['archive_name']
        archive_path = os.path.join(root, archive_name)
        if archive == 'dev':
            urls = _ARCHIVE_CONFIGS[archive]['urls']
            checksums = _ARCHIVE_CONFIGS[archive]['checksums']
            with open(archive_path, 'wb') as f:
                for url, checksum in zip(urls, checksums):
                    file_path = os.path.join(root, os.path.basename(url))
                    download_url_to_file(url, file_path, hash_prefix=checksum)
                    with open(file_path, 'rb') as f_split:
                        f.write(f_split.read())
        else:
            url = _ARCHIVE_CONFIGS[archive]['url']
            checksum = _ARCHIVE_CONFIGS[archive]['checksum']
            download_url_to_file(url, archive_path, hash_prefix=checksum)
        _extract_zip(archive_path)