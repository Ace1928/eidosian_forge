from __future__ import annotations
import os
import pathlib
from typing import TYPE_CHECKING
import numpy as np
from xarray.backends.api import open_dataset as _open_dataset
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
def _construct_cache_dir(path):
    import pooch
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    elif path is None:
        path = pooch.os_cache(_default_cache_dir_name)
    return path