import datetime
import io
import logging
import os
import os.path as osp
import re
import shutil
import stat
import tempfile
from fsspec import AbstractFileSystem
from fsspec.compression import compr
from fsspec.core import get_compression
from fsspec.utils import isfilelike, stringify_path
@classmethod
def _strip_protocol(cls, path):
    path = stringify_path(path)
    if path.startswith('file://'):
        path = path[7:]
    elif path.startswith('file:'):
        path = path[5:]
    elif path.startswith('local://'):
        path = path[8:]
    elif path.startswith('local:'):
        path = path[6:]
    return make_path_posix(path).rstrip('/') or cls.root_marker