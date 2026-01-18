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
@deprecated('Use gzip.GzipFile instead as it also uses a buffer.')
class BufferedGzipFile(GzipFile):
    """A ``GzipFile`` subclass for compatibility with older nltk releases.

    Use ``GzipFile`` directly as it also buffers in all supported
    Python versions.
    """

    @py3_data
    def __init__(self, filename=None, mode=None, compresslevel=9, fileobj=None, **kwargs):
        """Return a buffered gzip file object."""
        GzipFile.__init__(self, filename, mode, compresslevel, fileobj)

    def write(self, data):
        super().write(data)