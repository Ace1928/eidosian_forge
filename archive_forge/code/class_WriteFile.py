import sys, os, unicodedata
import py
from py.builtin import text, bytes
class WriteFile(object):

    def __init__(self, writemethod, encoding=None):
        self.encoding = encoding
        self._writemethod = writemethod

    def write(self, data):
        if self.encoding:
            data = data.encode(self.encoding, 'replace')
        self._writemethod(data)

    def flush(self):
        return