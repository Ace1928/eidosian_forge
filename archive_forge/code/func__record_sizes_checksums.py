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
def _record_sizes_checksums(self, url_or_urls: NestedDataStructure, downloaded_path_or_paths: NestedDataStructure):
    """Record size/checksum of downloaded files."""
    delay = 5
    for url, path in hf_tqdm(list(zip(url_or_urls.flatten(), downloaded_path_or_paths.flatten())), delay=delay, desc='Computing checksums'):
        self._recorded_sizes_checksums[str(url)] = get_size_checksum_dict(path, record_checksum=self.record_checksums)