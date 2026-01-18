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
def _glob_full(conf: Config, pattern: str) -> Iterator[DirEntry]:
    prefix, _, _ = pattern.partition('*')
    re_pattern = _compile_pattern(pattern)
    for entry in _expand_implicit_dirs(root=prefix, it=_list_blobs(conf=conf, path=prefix)):
        entry_slash_path = _get_slash_path(entry)
        if bool(re_pattern.match(entry_slash_path)):
            if entry_slash_path == prefix and entry.is_dir:
                continue
            yield entry