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
def _proc_sparse(self, tarfile):
    """Process a GNU sparse header plus extra headers.
        """
    structs, isextended, origsize = self._sparse_structs
    del self._sparse_structs
    while isextended:
        buf = tarfile.fileobj.read(BLOCKSIZE)
        pos = 0
        for i in range(21):
            try:
                offset = nti(buf[pos:pos + 12])
                numbytes = nti(buf[pos + 12:pos + 24])
            except ValueError:
                break
            if offset and numbytes:
                structs.append((offset, numbytes))
            pos += 24
        isextended = bool(buf[504])
    self.sparse = structs
    self.offset_data = tarfile.fileobj.tell()
    tarfile.offset = self.offset_data + self._block(self.size)
    self.size = origsize
    return self