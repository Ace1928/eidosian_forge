from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class PdfBinary:

    def __init__(self, data):
        self.data = data

    def __bytes__(self):
        return b'<%s>' % b''.join((b'%02X' % b for b in self.data))