import errno
import os
import sys
from stat import S_IMODE, S_ISDIR, ST_MODE
from .. import osutils, transport, urlutils
def _check_mode_and_size(self, file_abspath, fd, mode=None):
    """Check the mode of the file, and return the current size"""
    st = os.fstat(fd)
    if mode is not None and mode != S_IMODE(st.st_mode):
        osutils.chmod_if_possible(file_abspath, mode)
    return st.st_size