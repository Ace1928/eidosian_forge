import codecs
import functools
import os
import pickle
import re
import sys
import textwrap
import zipfile
from abc import ABCMeta, abstractmethod
from gzip import WRITE as GZ_WRITE
from gzip import GzipFile
from io import BytesIO, TextIOWrapper
from urllib.request import url2pathname, urlopen
from nltk import grammar, sem
from nltk.compat import add_py3_data, py3_data
from nltk.internals import deprecated
class FileSystemPathPointer(PathPointer, str):
    """
    A path pointer that identifies a file which can be accessed
    directly via a given absolute path.
    """

    @py3_data
    def __init__(self, _path):
        """
        Create a new path pointer for the given absolute path.

        :raise IOError: If the given path does not exist.
        """
        _path = os.path.abspath(_path)
        if not os.path.exists(_path):
            raise OSError('No such file or directory: %r' % _path)
        self._path = _path

    @property
    def path(self):
        """The absolute path identified by this path pointer."""
        return self._path

    def open(self, encoding=None):
        stream = open(self._path, 'rb')
        if encoding is not None:
            stream = SeekableUnicodeStreamReader(stream, encoding)
        return stream

    def file_size(self):
        return os.stat(self._path).st_size

    def join(self, fileid):
        _path = os.path.join(self._path, fileid)
        return FileSystemPathPointer(_path)

    def __repr__(self):
        return 'FileSystemPathPointer(%r)' % self._path

    def __str__(self):
        return self._path