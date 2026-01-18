import enum
import io
import os
import posixpath
import tarfile
import warnings
import zipfile
from datetime import datetime
from functools import partial
from itertools import chain
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
from .. import config
from ..utils import tqdm as hf_tqdm
from ..utils.deprecation_utils import DeprecatedEnum, deprecated
from ..utils.file_utils import (
from ..utils.info_utils import get_size_checksum_dict
from ..utils.logging import get_logger
from ..utils.py_utils import NestedDataStructure, map_nested, size_str
from ..utils.track import TrackedIterable, tracked_str
from .download_config import DownloadConfig
@classmethod
def _iter_from_paths(cls, urlpaths: Union[str, List[str]]) -> Generator[str, None, None]:
    if not isinstance(urlpaths, list):
        urlpaths = [urlpaths]
    for urlpath in urlpaths:
        if os.path.isfile(urlpath):
            yield urlpath
        else:
            for dirpath, dirnames, filenames in os.walk(urlpath):
                dirnames[:] = sorted([dirname for dirname in dirnames if not dirname.startswith(('.', '__'))])
                if os.path.basename(dirpath).startswith(('.', '__')):
                    continue
                for filename in sorted(filenames):
                    if filename.startswith(('.', '__')):
                        continue
                    yield os.path.join(dirpath, filename)