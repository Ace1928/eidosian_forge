import collections.abc
import contextlib
import datetime
import errno
import functools
import io
import os
import pathlib
import queue
import re
import stat
import sys
import time
from multiprocessing import Process
from threading import Thread
from typing import IO, Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Type, Union
import multivolumefile
from py7zr.archiveinfo import Folder, Header, SignatureHeader
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods, get_methods_names
from py7zr.exceptions import (
from py7zr.helpers import (
from py7zr.properties import DEFAULT_FILTERS, FILTER_DEFLATE64, MAGIC_7Z, get_default_blocksize, get_memory_limit
def _writeall(self, path, arcname):
    try:
        if path.is_symlink() and (not self.dereference):
            self.write(path, arcname)
        elif path.is_file():
            self.write(path, arcname)
        elif path.is_dir():
            if not path.samefile('.'):
                self.write(path, arcname)
            for nm in sorted(os.listdir(str(path))):
                arc = os.path.join(arcname, nm) if arcname is not None else None
                self._writeall(path.joinpath(nm), arc)
        else:
            return
    except OSError as ose:
        if self.dereference and ose.errno in [errno.ELOOP]:
            return
        elif self.dereference and sys.platform == 'win32' and (ose.errno in [errno.ENOENT]):
            return
        else:
            raise