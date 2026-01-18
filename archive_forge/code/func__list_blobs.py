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
def _list_blobs(conf: Config, path: str, delimiter: Optional[str]=None) -> Iterator[DirEntry]:
    params = {}
    if delimiter is not None:
        params['delimiter'] = delimiter
    if _is_gcp_path(path):
        yield from gcp.list_blobs(conf, path, delimiter=delimiter)
    elif _is_azure_path(path):
        yield from azure.list_blobs(conf, path, delimiter=delimiter)
    else:
        raise Error(f"Unrecognized path: '{path}'")