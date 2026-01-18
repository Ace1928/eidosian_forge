import os
import re
from functools import partial
from glob import has_magic
from pathlib import Path, PurePath
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import huggingface_hub
from fsspec import get_fs_token_paths
from fsspec.implementations.http import HTTPFileSystem
from huggingface_hub import HfFileSystem
from packaging import version
from tqdm.contrib.concurrent import thread_map
from . import config
from .download import DownloadConfig
from .download.streaming_download_manager import _prepare_path_and_storage_options, xbasename, xjoin
from .naming import _split_re
from .splits import Split
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import is_local_path, is_relative_path
from .utils.py_utils import glob_pattern_to_regex, string_to_dict
def _get_origin_metadata(data_files: List[str], max_workers=64, download_config: Optional[DownloadConfig]=None) -> Tuple[str]:
    return thread_map(partial(_get_single_origin_metadata, download_config=download_config), data_files, max_workers=max_workers, tqdm_class=hf_tqdm, desc='Resolving data files', disable=len(data_files) <= 16 or None)