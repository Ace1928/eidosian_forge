from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class PdfStream:

    def __init__(self, dictionary, buf):
        self.dictionary = dictionary
        self.buf = buf

    def decode(self):
        try:
            filter = self.dictionary.Filter
        except AttributeError:
            return self.buf
        if filter == b'FlateDecode':
            try:
                expected_length = self.dictionary.DL
            except AttributeError:
                expected_length = self.dictionary.Length
            return zlib.decompress(self.buf, bufsize=int(expected_length))
        else:
            msg = f'stream filter {repr(self.dictionary.Filter)} unknown/unsupported'
            raise NotImplementedError(msg)