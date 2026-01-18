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
def _prepare_write(self, filters, password):
    if password is not None and filters is None:
        filters = DEFAULT_FILTERS.ENCRYPTED_ARCHIVE_FILTER
    elif filters is None:
        filters = DEFAULT_FILTERS.ARCHIVE_FILTER
    self.files = ArchiveFileList()
    self.sig_header = SignatureHeader()
    self.sig_header._write_skelton(self.fp)
    self.afterheader = self.fp.tell()
    self.header = Header.build_header(filters, password)
    self.fp.seek(self.afterheader)
    self.worker = Worker(self.files, self.afterheader, self.header, self.mp)