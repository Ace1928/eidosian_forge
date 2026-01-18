import typing
import urllib.parse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import List, Optional, Union
from requests import get
from pennylane.data.base import Dataset
from pennylane.data.base.hdf5 import open_hdf5_s3
from .foldermap import DataPath, FolderMapView, ParamArg
from .params import DEFAULT, FULL, format_params
def _download_partial(s3_url: str, dest: Path, attributes: Optional[typing.Iterable[str]], overwrite: bool, block_size: int) -> None:
    """Download the requested attributes of the Dataset at ``s3_path`` into ``dest``.

    Args:
        s3_url: URL of the remote dataset
        dest: Destination dataset path
        attributes: Requested attributes to download. Passing ``None`` is equivalent
            to requesting all attributes of the remote dataset.
        overwrite: If True, overwrite attributes that already exist at ``dest``. Otherwise,
            only download attributes that do not exist at ``dest``.
    """
    dest_dataset = Dataset.open(dest, mode='a')
    remote_dataset = None
    attributes_to_fetch = set()
    if attributes is not None:
        attributes_to_fetch.update(attributes)
    else:
        remote_dataset = Dataset(open_hdf5_s3(s3_url, block_size=block_size))
        attributes_to_fetch.update(remote_dataset.attrs)
    if not overwrite:
        attributes_to_fetch.difference_update(dest_dataset.attrs)
    if len(attributes_to_fetch) > 0:
        remote_dataset = remote_dataset or Dataset(open_hdf5_s3(s3_url, block_size=block_size))
        remote_dataset.write(dest_dataset, 'a', attributes, overwrite=overwrite)
    if remote_dataset:
        remote_dataset.close()
    dest_dataset.close()
    del remote_dataset
    del dest_dataset