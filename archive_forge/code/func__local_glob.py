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
def _local_glob(pattern: str) -> Iterator[str]:
    normalized_pattern = os.path.normpath(pattern)
    if pattern.endswith('/') or pattern.endswith('\\'):
        normalized_pattern += os.sep
    if '*' in normalized_pattern:
        prefix = normalized_pattern.split('*')[0]
        if prefix.endswith(os.sep):
            base_dir = os.path.abspath(prefix)
            if not base_dir.endswith(os.sep):
                base_dir += os.sep
            pattern_suffix = normalized_pattern[len(prefix):]
        else:
            dirpath = os.path.dirname(prefix)
            base_dir = os.path.abspath(dirpath)
            if len(dirpath) == 0:
                pattern_suffix = os.sep + normalized_pattern
            else:
                pattern_suffix = normalized_pattern[len(dirpath):]
        full_pattern = base_dir + pattern_suffix
        regexp = _compile_pattern(full_pattern, sep=os.sep)
        for root, dirnames, filenames in os.walk(base_dir):
            paths = [os.path.join(root, dirname + os.sep) for dirname in dirnames]
            paths += [os.path.join(root, filename) for filename in filenames]
            for path in paths:
                if re.match(regexp, path):
                    if path.endswith(os.sep):
                        path = path[:-1]
                    yield path
    else:
        path = os.path.abspath(pattern)
        if os.path.exists(path):
            if path.endswith(os.sep):
                path = path[:-1]
            yield path