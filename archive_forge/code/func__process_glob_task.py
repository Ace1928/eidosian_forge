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
def _process_glob_task(conf: Config, root: str, t: _GlobTask) -> Iterator[Union[_GlobTask, _GlobEntry]]:
    cur = t.cur + t.rem[0]
    rem = t.rem[1:]
    if '**' in cur:
        for entry in _glob_full(conf, root + cur + ''.join(rem)):
            yield _GlobEntry(entry)
    elif '*' in cur:
        re_pattern = _compile_pattern(root + cur)
        prefix, _, _ = cur.partition('*')
        path = root + prefix
        for entry in _list_blobs(conf=conf, path=path, delimiter='/'):
            entry_slash_path = _get_slash_path(entry)
            if entry_slash_path == path and entry.is_dir:
                continue
            if bool(re_pattern.match(entry_slash_path)):
                if len(rem) == 0:
                    yield _GlobEntry(entry)
                else:
                    assert entry_slash_path.startswith(root)
                    yield _GlobTask(entry_slash_path[len(root):], rem)
    elif len(rem) == 0:
        path = root + cur
        entry = _get_entry(conf, path)
        if entry is not None:
            yield _GlobEntry(entry)
    else:
        yield _GlobTask(cur, rem)