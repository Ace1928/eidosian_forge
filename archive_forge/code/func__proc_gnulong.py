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
def _proc_gnulong(self, tarfile):
    """Process the blocks that hold a GNU longname
           or longlink member.
        """
    buf = tarfile.fileobj.read(self._block(self.size))
    try:
        next = self.fromtarfile(tarfile)
    except HeaderError as e:
        raise SubsequentHeaderError(str(e)) from None
    next.offset = self.offset
    if self.type == GNUTYPE_LONGNAME:
        next.name = nts(buf, tarfile.encoding, tarfile.errors)
    elif self.type == GNUTYPE_LONGLINK:
        next.linkname = nts(buf, tarfile.encoding, tarfile.errors)
    if next.isdir():
        next.name = next.name.removesuffix('/')
    return next