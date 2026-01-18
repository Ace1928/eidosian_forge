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
def _fetch_range(self, start, end):
    if 'r' not in self.mode:
        raise ValueError
    self._open()
    self.f.seek(start)
    return self.f.read(end - start)