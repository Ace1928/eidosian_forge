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
def _proc_builtin(self, tarfile):
    """Process a builtin type or an unknown type which
           will be treated as a regular file.
        """
    self.offset_data = tarfile.fileobj.tell()
    offset = self.offset_data
    if self.isreg() or self.type not in SUPPORTED_TYPES:
        offset += self._block(self.size)
    tarfile.offset = offset
    self._apply_pax_info(tarfile.pax_headers, tarfile.encoding, tarfile.errors)
    if self.isdir():
        self.name = self.name.rstrip('/')
    return self