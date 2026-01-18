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
def _init_write_gz(self):
    """Initialize for writing with gzip compression.
        """
    self.cmp = self.zlib.compressobj(9, self.zlib.DEFLATED, -self.zlib.MAX_WBITS, self.zlib.DEF_MEM_LEVEL, 0)
    timestamp = struct.pack('<L', int(time.time()))
    self.__write(b'\x1f\x8b\x08\x08' + timestamp + b'\x02\xff')
    if self.name.endswith('.gz'):
        self.name = self.name[:-3]
    self.name = os.path.basename(self.name)
    self.__write(self.name.encode('iso-8859-1', 'replace') + NUL)