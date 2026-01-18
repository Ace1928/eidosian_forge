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
def _make_file_info_from_name(self, bio, size: int, arcname: str) -> Dict[str, Any]:
    f: Dict[str, Any] = {}
    f['origin'] = None
    f['data'] = bio
    f['filename'] = pathlib.Path(arcname).as_posix()
    f['uncompressed'] = size
    f['emptystream'] = size == 0
    f['attributes'] = getattr(stat, 'FILE_ATTRIBUTE_ARCHIVE')
    f['creationtime'] = ArchiveTimestamp.from_now()
    f['lastwritetime'] = ArchiveTimestamp.from_now()
    return f