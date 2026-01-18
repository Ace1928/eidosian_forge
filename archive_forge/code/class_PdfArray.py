from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class PdfArray(list):

    def __bytes__(self):
        return b'[ ' + b' '.join((pdf_repr(x) for x in self)) + b' ]'