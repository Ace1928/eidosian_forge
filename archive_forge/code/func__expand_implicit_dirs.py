from __future__ import annotations
import binascii
import collections
import concurrent.futures
import contextlib
import hashlib
import io
import itertools
import math
import multiprocessing as mp
import os
import re
import shutil
import stat as stat_module
import tempfile
import time
import urllib.parse
from functools import partial
from types import ModuleType
from typing import (
import urllib3
import filelock
from blobfile import _azure as azure
from blobfile import _common as common
from blobfile import _gcp as gcp
from blobfile._common import (
def _expand_implicit_dirs(root: str, it: Iterator[DirEntry]) -> Iterator[DirEntry]:
    if _is_gcp_path(root):
        entry_from_dirpath = gcp.entry_from_dirpath
    elif _is_azure_path(root):
        entry_from_dirpath = azure.entry_from_dirpath
    else:
        raise Error(f"Unrecognized path '{root}'")
    previous_path = root
    for entry in it:
        entry_slash_path = _get_slash_path(entry)
        offset = _string_overlap(previous_path, entry_slash_path)
        relpath = entry_slash_path[offset:]
        cur = entry_slash_path[:offset]
        for part in _split_path(relpath)[:-1]:
            cur += part
            yield entry_from_dirpath(cur)
        yield entry
        assert entry_slash_path >= previous_path
        previous_path = entry_slash_path