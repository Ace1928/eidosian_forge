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
class GzipFileSystemPathPointer(FileSystemPathPointer):
    """
    A subclass of ``FileSystemPathPointer`` that identifies a gzip-compressed
    file located at a given absolute path.  ``GzipFileSystemPathPointer`` is
    appropriate for loading large gzip-compressed pickle objects efficiently.
    """

    def open(self, encoding=None):
        stream = GzipFile(self._path, 'rb')
        if encoding:
            stream = SeekableUnicodeStreamReader(stream, encoding)
        return stream