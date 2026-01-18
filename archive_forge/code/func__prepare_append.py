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
def _prepare_append(self, filters, password):
    if password is not None and filters is None:
        filters = DEFAULT_FILTERS.ENCRYPTED_ARCHIVE_FILTER
    elif filters is None:
        filters = DEFAULT_FILTERS.ARCHIVE_FILTER
    else:
        for f in filters:
            if f['id'] == FILTER_DEFLATE64:
                raise UnsupportedCompressionMethodError(filters, 'Compression with deflate64 is not supported.')
    self.header.filters = filters
    self.header.password = password
    if self.header.main_streams is not None:
        pos = self.afterheader + self.header.main_streams.packinfo.packpositions[-1]
    else:
        pos = self.afterheader
    self.fp.seek(pos)
    self.worker = Worker(self.files, pos, self.header, self.mp)