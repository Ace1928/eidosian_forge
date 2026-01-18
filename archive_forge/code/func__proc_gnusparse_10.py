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
def _proc_gnusparse_10(self, next, pax_headers, tarfile):
    """Process a GNU tar extended sparse header, version 1.0.
        """
    fields = None
    sparse = []
    buf = tarfile.fileobj.read(BLOCKSIZE)
    fields, buf = buf.split(b'\n', 1)
    fields = int(fields)
    while len(sparse) < fields * 2:
        if b'\n' not in buf:
            buf += tarfile.fileobj.read(BLOCKSIZE)
        number, buf = buf.split(b'\n', 1)
        sparse.append(int(number))
    next.offset_data = tarfile.fileobj.tell()
    next.sparse = list(zip(sparse[::2], sparse[1::2]))