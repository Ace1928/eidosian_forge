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
def _download_full(s3_url: str, dest: Path):
    """Download the full dataset file at ``s3_url`` to ``path``."""
    with open(dest, 'wb') as f:
        resp = get(s3_url, timeout=5.0)
        resp.raise_for_status()
        f.write(resp.content)