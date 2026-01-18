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
def _handle_fatal_error(self, e):
    """Handle "fatal" error according to self.errorlevel"""
    if self.errorlevel > 0:
        raise
    elif isinstance(e, OSError):
        if e.filename is None:
            self._dbg(1, 'tarfile: %s' % e.strerror)
        else:
            self._dbg(1, 'tarfile: %s %r' % (e.strerror, e.filename))
    else:
        self._dbg(1, 'tarfile: %s %s' % (type(e).__name__, e))