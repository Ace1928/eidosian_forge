import io
import logging
from pathlib import Path
from typing import IO, Any, Dict, Union
import fsspec
import fsspec.utils
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from lightning_utilities.core.imports import module_available
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
def _is_local_file_protocol(path: _PATH) -> bool:
    return fsspec.utils.get_protocol(str(path)) == 'file'