import atexit
import logging
import os
import time
from concurrent.futures import Future
from dataclasses import dataclass
from io import SEEK_END, SEEK_SET, BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Union
from .hf_api import IGNORE_GIT_FOLDER_PATTERNS, CommitInfo, CommitOperationAdd, HfApi
from .utils import filter_repo_objects
@dataclass(frozen=True)
class _FileToUpload:
    """Temporary dataclass to store info about files to upload. Not meant to be used directly."""
    local_path: Path
    path_in_repo: str
    size_limit: int
    last_modified: float