import io
import os
from typing import Iterable, List, Optional, Tuple, Union
from ..utils.file_utils import (  # noqa: F401 # backward compatibility
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig
def _download_single(self, urlpath: str) -> str:
    urlpath = str(urlpath)
    if is_relative_path(urlpath):
        urlpath = url_or_path_join(self._base_path, urlpath)
    return urlpath