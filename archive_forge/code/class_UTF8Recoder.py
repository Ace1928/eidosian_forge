from __future__ import absolute_import, print_function, division
import csv
import cStringIO
from petl.util.base import Table, data
from petl.io.base import getcodec
class UTF8Recoder:

    def __init__(self, buf, encoding, errors):
        codec = getcodec(encoding)
        self.reader = codec.streamreader(buf, errors=errors)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode('utf-8')