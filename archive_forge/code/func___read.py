from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def __read(self, size):
    """Return size bytes from stream. If internal buffer is empty,
           read another block from the stream.
        """
    c = len(self.buf)
    t = [self.buf]
    while c < size:
        buf = self.fileobj.read(self.bufsize)
        if not buf:
            break
        t.append(buf)
        c += len(buf)
    t = b''.join(t)
    self.buf = t[size:]
    return t[:size]