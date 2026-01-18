import os
from io import BytesIO
import zipfile
import tempfile
import shutil
import enum
import warnings
from ..core import urlopen, get_remote_file
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
def _read_first_bytes(self, N=256):
    if self._bytes is not None:
        self._firstbytes = self._bytes[:N]
    else:
        try:
            f = self.get_file()
        except IOError:
            if os.path.isdir(self.filename):
                self._firstbytes = bytes()
                return
            raise
        try:
            i = f.tell()
        except Exception:
            i = None
        self._firstbytes = read_n_bytes(f, N)
        try:
            if i is None:
                raise Exception('cannot seek with None')
            f.seek(i)
        except Exception:
            self._file = None
            if self._uri_type == URI_FILE:
                raise IOError('Cannot seek back after getting firstbytes!')