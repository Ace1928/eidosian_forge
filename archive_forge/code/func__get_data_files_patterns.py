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
def _get_data_files_patterns(pattern_resolver: Callable[[str], List[str]]) -> Dict[str, List[str]]:
    """
    Get the default pattern from a directory or repository by testing all the supported patterns.
    The first patterns to return a non-empty list of data files is returned.

    In order, it first tests if SPLIT_PATTERN_SHARDED works, otherwise it tests the patterns in ALL_DEFAULT_PATTERNS.
    """
    for split_pattern in ALL_SPLIT_PATTERNS:
        pattern = split_pattern.replace('{split}', '*')
        try:
            data_files = pattern_resolver(pattern)
        except FileNotFoundError:
            continue
        if len(data_files) > 0:
            splits: Set[str] = {string_to_dict(xbasename(p), glob_pattern_to_regex(xbasename(split_pattern)))['split'] for p in data_files}
            if any((not re.match(_split_re, split) for split in splits)):
                raise ValueError(f"Split name should match '{_split_re}'' but got '{splits}'.")
            sorted_splits = [str(split) for split in DEFAULT_SPLITS if split in splits] + sorted(splits - set(DEFAULT_SPLITS))
            return {split: [split_pattern.format(split=split)] for split in sorted_splits}
    for patterns_dict in ALL_DEFAULT_PATTERNS:
        non_empty_splits = []
        for split, patterns in patterns_dict.items():
            for pattern in patterns:
                try:
                    data_files = pattern_resolver(pattern)
                except FileNotFoundError:
                    continue
                if len(data_files) > 0:
                    non_empty_splits.append(split)
                    break
        if non_empty_splits:
            return {split: patterns_dict[split] for split in non_empty_splits}
    raise FileNotFoundError(f"Couldn't resolve pattern {pattern} with resolver {pattern_resolver}")