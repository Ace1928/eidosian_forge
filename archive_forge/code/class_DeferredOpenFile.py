import random
import time
import functools
import math
import os
import socket
import stat
import string
import logging
import threading
import io
from collections import defaultdict
from botocore.exceptions import IncompleteReadError
from botocore.exceptions import ReadTimeoutError
from s3transfer.compat import SOCKET_ERROR
from s3transfer.compat import rename_file
from s3transfer.compat import seekable
from s3transfer.compat import fallocate
class DeferredOpenFile(object):

    def __init__(self, filename, start_byte=0, mode='rb', open_function=open):
        """A class that defers the opening of a file till needed

        This is useful for deferring opening of a file till it is needed
        in a separate thread, as there is a limit of how many open files
        there can be in a single thread for most operating systems. The
        file gets opened in the following methods: ``read()``, ``seek()``,
        and ``__enter__()``

        :type filename: str
        :param filename: The name of the file to open

        :type start_byte: int
        :param start_byte: The byte to seek to when the file is opened.

        :type mode: str
        :param mode: The mode to use to open the file

        :type open_function: function
        :param open_function: The function to use to open the file
        """
        self._filename = filename
        self._fileobj = None
        self._start_byte = start_byte
        self._mode = mode
        self._open_function = open_function

    def _open_if_needed(self):
        if self._fileobj is None:
            self._fileobj = self._open_function(self._filename, self._mode)
            if self._start_byte != 0:
                self._fileobj.seek(self._start_byte)

    @property
    def name(self):
        return self._filename

    def read(self, amount=None):
        self._open_if_needed()
        return self._fileobj.read(amount)

    def write(self, data):
        self._open_if_needed()
        self._fileobj.write(data)

    def seek(self, where):
        self._open_if_needed()
        self._fileobj.seek(where)

    def tell(self):
        if self._fileobj is None:
            return self._start_byte
        return self._fileobj.tell()

    def close(self):
        if self._fileobj:
            self._fileobj.close()

    def __enter__(self):
        self._open_if_needed()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()