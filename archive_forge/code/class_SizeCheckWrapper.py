import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
class SizeCheckWrapper:
    """Wraps a file-like object, raising MaxSizeExceeded if too large.

    :param rfile: ``file`` of a limited size
    :param int maxlen: maximum length of the file being read
    """

    def __init__(self, rfile, maxlen):
        """Initialize SizeCheckWrapper instance."""
        self.rfile = rfile
        self.maxlen = maxlen
        self.bytes_read = 0

    def _check_length(self):
        if self.maxlen and self.bytes_read > self.maxlen:
            raise errors.MaxSizeExceeded()

    def read(self, size=None):
        """Read a chunk from ``rfile`` buffer and return it.

        :param size: amount of data to read
        :type size: int

        :returns: chunk from ``rfile``, limited by size if specified
        :rtype: bytes
        """
        data = self.rfile.read(size)
        self.bytes_read += len(data)
        self._check_length()
        return data

    def readline(self, size=None):
        """Read a single line from ``rfile`` buffer and return it.

        :param size: minimum amount of data to read
        :type size: int

        :returns: one line from ``rfile``
        :rtype: bytes
        """
        if size is not None:
            data = self.rfile.readline(size)
            self.bytes_read += len(data)
            self._check_length()
            return data
        res = []
        while True:
            data = self.rfile.readline(256)
            self.bytes_read += len(data)
            self._check_length()
            res.append(data)
            if len(data) < 256 or data[-1:] == LF:
                return EMPTY.join(res)

    def readlines(self, sizehint=0):
        """Read all lines from ``rfile`` buffer and return them.

        :param sizehint: hint of minimum amount of data to read
        :type sizehint: int

        :returns: lines of bytes read from ``rfile``
        :rtype: list[bytes]
        """
        total = 0
        lines = []
        line = self.readline(sizehint)
        while line:
            lines.append(line)
            total += len(line)
            if 0 < sizehint <= total:
                break
            line = self.readline(sizehint)
        return lines

    def close(self):
        """Release resources allocated for ``rfile``."""
        self.rfile.close()

    def __iter__(self):
        """Return file iterator."""
        return self

    def __next__(self):
        """Generate next file chunk."""
        data = next(self.rfile)
        self.bytes_read += len(data)
        self._check_length()
        return data
    next = __next__