from __future__ import division
import re
import stat
from .helpers import (
class BlobCommand(ImportCommand):

    def __init__(self, mark, data, lineno=0):
        ImportCommand.__init__(self, b'blob')
        self.mark = mark
        self.data = data
        self.lineno = lineno
        if mark is None:
            self.id = b'@' + ('%d' % lineno).encode('utf-8')
        else:
            self.id = b':' + mark
        self._binary = [b'data']

    def __bytes__(self):
        if self.mark is None:
            mark_line = b''
        else:
            mark_line = b'\nmark :' + self.mark
        return b'blob' + mark_line + b'\n' + ('data %d\n' % len(self.data)).encode('utf-8') + self.data