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
def _glob_worker(conf: Config, root: str, tasks: mp.Queue[_GlobTask], results: mp.Queue[Union[_GlobEntry, _GlobTask, _GlobTaskComplete]]) -> None:
    while True:
        t = tasks.get()
        for r in _process_glob_task(conf, root=root, t=t):
            results.put(r)
        results.put(_GlobTaskComplete())