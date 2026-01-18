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
def _sharded_listdir_worker(conf: Config, prefixes: mp.Queue[Tuple[str, str, bool]], items: mp.Queue[Optional[DirEntry]]) -> None:
    while True:
        base, prefix, exact = prefixes.get(True)
        if exact:
            path = base + prefix
            entry = _get_entry(conf, path)
            if entry is not None:
                items.put(entry)
        else:
            it = _list_blobs_in_dir(conf, base + prefix, exclude_prefix=False)
            for entry in it:
                items.put(entry)
        items.put(None)