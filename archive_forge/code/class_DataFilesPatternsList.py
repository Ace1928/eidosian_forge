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
class DataFilesPatternsList(List[str]):
    """
    List of data files patterns (absolute local paths or URLs).
    For each pattern there should also be a list of allowed extensions
    to keep, or a None ot keep all the files for the pattern.
    """

    def __init__(self, patterns: List[str], allowed_extensions: List[Optional[List[str]]]):
        super().__init__(patterns)
        self.allowed_extensions = allowed_extensions

    def __add__(self, other):
        return DataFilesList([*self, *other], self.allowed_extensions + other.allowed_extensions)

    @classmethod
    def from_patterns(cls, patterns: List[str], allowed_extensions: Optional[List[str]]=None) -> 'DataFilesPatternsDict':
        return cls(patterns, [allowed_extensions] * len(patterns))

    def resolve(self, base_path: str, download_config: Optional[DownloadConfig]=None) -> 'DataFilesList':
        base_path = base_path if base_path is not None else Path().resolve().as_posix()
        data_files = []
        for pattern, allowed_extensions in zip(self, self.allowed_extensions):
            try:
                data_files.extend(resolve_pattern(pattern, base_path=base_path, allowed_extensions=allowed_extensions, download_config=download_config))
            except FileNotFoundError:
                if not has_magic(pattern):
                    raise
        origin_metadata = _get_origin_metadata(data_files, download_config=download_config)
        return DataFilesList(data_files, origin_metadata)

    def filter_extensions(self, extensions: List[str]) -> 'DataFilesList':
        return DataFilesPatternsList(self, [allowed_extensions + extensions for allowed_extensions in self.allowed_extensions])