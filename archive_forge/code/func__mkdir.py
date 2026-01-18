import errno
import os
import sys
from stat import S_IMODE, S_ISDIR, ST_MODE
from .. import osutils, transport, urlutils
def _mkdir(self, abspath, mode=None):
    """Create a real directory, filtering through mode"""
    if mode is None:
        local_mode = 511
    else:
        local_mode = mode
    try:
        os.mkdir(abspath, local_mode)
    except OSError as e:
        self._translate_error(e, abspath)
    if mode is not None:
        try:
            osutils.chmod_if_possible(abspath, mode)
        except OSError as e:
            self._translate_error(e, abspath)