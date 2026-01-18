import glob
import io
import os
import posixpath
import re
import tarfile
import time
import xml.dom.minidom
import zipfile
from asyncio import TimeoutError
from io import BytesIO
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET
import fsspec
from aiohttp.client_exceptions import ClientError
from huggingface_hub.utils import EntryNotFoundError
from packaging import version
from .. import config
from ..filesystems import COMPRESSION_FILESYSTEMS
from ..utils.file_utils import (
from ..utils.logging import get_logger
from ..utils.py_utils import map_nested
from .download_config import DownloadConfig
def _get_extraction_protocol(urlpath: str, download_config: Optional[DownloadConfig]=None) -> Optional[str]:
    urlpath = str(urlpath)
    path = urlpath.split('::')[0]
    extension = _get_path_extension(path)
    if extension in BASE_KNOWN_EXTENSIONS or extension in ['tgz', 'tar'] or path.endswith(('.tar.gz', '.tar.bz2', '.tar.xz')):
        return None
    elif extension in COMPRESSION_EXTENSION_TO_PROTOCOL:
        return COMPRESSION_EXTENSION_TO_PROTOCOL[extension]
    urlpath, storage_options = _prepare_path_and_storage_options(urlpath, download_config=download_config)
    try:
        with fsspec.open(urlpath, **storage_options or {}) as f:
            return _get_extraction_protocol_with_magic_number(f)
    except FileNotFoundError:
        if urlpath.startswith(config.HF_ENDPOINT):
            raise FileNotFoundError(urlpath + '\nIf the repo is private or gated, make sure to log in with `huggingface-cli login`.') from None
        else:
            raise