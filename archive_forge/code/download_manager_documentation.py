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
Download and extract given `url_or_urls`.

        Is roughly equivalent to:

        ```
        extracted_paths = dl_manager.extract(dl_manager.download(url_or_urls))
        ```

        Args:
            url_or_urls (`str` or `list` or `dict`):
                URL or `list` or `dict` of URLs to download and extract. Each URL is a `str`.

        Returns:
            extracted_path(s): `str`, extracted paths of given URL(s).
        