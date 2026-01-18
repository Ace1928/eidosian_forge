import inspect
import io
import os
import re
import warnings
from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import ceil
from os.path import getsize
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional, Tuple, TypedDict
from urllib.parse import unquote
from huggingface_hub.constants import ENDPOINT, HF_HUB_ENABLE_HF_TRANSFER, REPO_TYPES_URL_PREFIXES
from huggingface_hub.utils import get_session
from .utils import (
from .utils.sha import sha256, sha_fileobj
def _validate_batch_error(lfs_batch_error: dict):
    """validates response from the LFS batch endpoint"""
    if not (isinstance(lfs_batch_error.get('oid'), str) and isinstance(lfs_batch_error.get('size'), int)):
        raise ValueError('lfs_batch_error is improperly formatted')
    error_info = lfs_batch_error.get('error')
    if not (isinstance(error_info, dict) and isinstance(error_info.get('message'), str) and isinstance(error_info.get('code'), int)):
        raise ValueError('lfs_batch_error is improperly formatted')
    return lfs_batch_error