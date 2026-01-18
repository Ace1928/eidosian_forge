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
def _writef(self, bio: IO[Any], arcname: str):
    if isinstance(bio, io.BytesIO):
        size = bio.getbuffer().nbytes
    elif isinstance(bio, io.TextIOBase):
        raise ValueError('Unsupported file object type: please open file with Binary mode.')
    elif isinstance(bio, io.BufferedIOBase):
        current = bio.tell()
        bio.seek(0, os.SEEK_END)
        last = bio.tell()
        bio.seek(current, os.SEEK_SET)
        size = last - current
    else:
        raise ValueError('Wrong argument passed for argument bio.')
    if size > 0:
        folder = self.header.initialize()
        file_info = self._make_file_info_from_name(bio, size, arcname)
        self.header.files_info.files.append(file_info)
        self.header.files_info.emptyfiles.append(file_info['emptystream'])
        self.files.append(file_info)
        self.worker.archive(self.fp, self.files, folder, deref=False)
    else:
        file_info = self._make_file_info_from_name(bio, size, arcname)
        self.header.files_info.files.append(file_info)
        self.header.files_info.emptyfiles.append(file_info['emptystream'])
        self.files.append(file_info)