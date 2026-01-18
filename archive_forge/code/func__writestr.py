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
def _writestr(self, data: Union[str, bytes, bytearray, memoryview], arcname: str):
    if not isinstance(arcname, str):
        raise ValueError('Unsupported arcname')
    if isinstance(data, str):
        self._writef(io.BytesIO(data.encode('UTF-8')), arcname)
    elif isinstance(data, bytes) or isinstance(data, bytearray) or isinstance(data, memoryview):
        self._writef(io.BytesIO(bytes(data)), arcname)
    else:
        raise ValueError('Unsupported data type.')