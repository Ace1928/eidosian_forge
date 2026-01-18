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
def _parallel_download(conf: Config, executor: concurrent.futures.Executor, src: str, dst: str, return_md5: bool) -> Optional[str]:
    ctx = Context(conf=conf)
    s = ctx.stat(src)
    if os.path.dirname(dst) != '':
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'wb') as f:
        if s.size > 0:
            f.seek(s.size - 1)
            f.write(b'\x00')
    max_workers = getattr(executor, '_max_workers', os.cpu_count() or 1)
    part_size = max(math.ceil(s.size / max_workers), common.PARALLEL_COPY_MINIMUM_PART_SIZE)
    start = 0
    futures = []
    while start < s.size:
        future = executor.submit(_download_chunk, conf, src, dst, start, min(part_size, s.size - start), s.size)
        futures.append(future)
        start += part_size
    for future in futures:
        future.result()
    if return_md5:
        with ctx.BlobFile(dst, 'rb') as f:
            return binascii.hexlify(common.block_md5(f)).decode('utf8')