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
def _char_seek_forward(self, offset, est_bytes=None):
    """
        Move the file position forward by ``offset`` characters,
        ignoring all buffers.

        :param est_bytes: A hint, giving an estimate of the number of
            bytes that will be needed to move forward by ``offset`` chars.
            Defaults to ``offset``.
        """
    if est_bytes is None:
        est_bytes = offset
    bytes = b''
    while True:
        newbytes = self.stream.read(est_bytes - len(bytes))
        bytes += newbytes
        chars, bytes_decoded = self._incr_decode(bytes)
        if len(chars) == offset:
            self.stream.seek(-len(bytes) + bytes_decoded, 1)
            return
        if len(chars) > offset:
            while len(chars) > offset:
                est_bytes += offset - len(chars)
                chars, bytes_decoded = self._incr_decode(bytes[:est_bytes])
            self.stream.seek(-len(bytes) + bytes_decoded, 1)
            return
        est_bytes += offset - len(chars)