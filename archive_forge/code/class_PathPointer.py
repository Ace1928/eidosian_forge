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
class PathPointer(metaclass=ABCMeta):
    """
    An abstract base class for 'path pointers,' used by NLTK's data
    package to identify specific paths.  Two subclasses exist:
    ``FileSystemPathPointer`` identifies a file that can be accessed
    directly via a given absolute path.  ``ZipFilePathPointer``
    identifies a file contained within a zipfile, that can be accessed
    by reading that zipfile.
    """

    @abstractmethod
    def open(self, encoding=None):
        """
        Return a seekable read-only stream that can be used to read
        the contents of the file identified by this path pointer.

        :raise IOError: If the path specified by this pointer does
            not contain a readable file.
        """

    @abstractmethod
    def file_size(self):
        """
        Return the size of the file pointed to by this path pointer,
        in bytes.

        :raise IOError: If the path specified by this pointer does
            not contain a readable file.
        """

    @abstractmethod
    def join(self, fileid):
        """
        Return a new path pointer formed by starting at the path
        identified by this pointer, and then following the relative
        path given by ``fileid``.  The path components of ``fileid``
        should be separated by forward slashes, regardless of
        the underlying file system's path separator character.
        """