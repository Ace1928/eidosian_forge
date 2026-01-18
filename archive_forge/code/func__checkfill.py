import sys, os, unicodedata
import py
from py.builtin import text, bytes
def _checkfill(self, line):
    diff2last = self._lastlen - len(line)
    if diff2last > 0:
        self.write(' ' * diff2last)