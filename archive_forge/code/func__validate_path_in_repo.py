import base64
import io
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union
from tqdm.contrib.concurrent import thread_map
from huggingface_hub import get_session
from .constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER
from .file_download import hf_hub_url
from .lfs import UploadInfo, lfs_upload, post_lfs_batch_info
from .utils import (
from .utils import tqdm as hf_tqdm
def _validate_path_in_repo(path_in_repo: str) -> str:
    if path_in_repo.startswith('/'):
        path_in_repo = path_in_repo[1:]
    if path_in_repo == '.' or path_in_repo == '..' or path_in_repo.startswith('../'):
        raise ValueError(f"Invalid `path_in_repo` in CommitOperation: '{path_in_repo}'")
    if path_in_repo.startswith('./'):
        path_in_repo = path_in_repo[2:]
    if any((part == '.git' for part in path_in_repo.split('/'))):
        raise ValueError(f"Invalid `path_in_repo` in CommitOperation: cannot update files under a '.git/' folder (path: '{path_in_repo}').")
    return path_in_repo