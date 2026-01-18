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
def _sanitize_archive_arcname(self, arcname):
    if isinstance(arcname, str):
        path = arcname
    else:
        path = str(arcname)
    if path.startswith(('/', os.sep)):
        path = path.lstrip('/' + os.sep)
    if re.match('^[a-zA-Z]:', path):
        path = path[2:]
        if path.startswith(('/', os.sep)):
            path = path.lstrip('/' + os.sep)
    if os.path.isabs(path) or re.match('^[a-zA-Z]:', path):
        raise AbsolutePathError(arcname)
    return path