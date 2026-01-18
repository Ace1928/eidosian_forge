from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
def _compute_filename_with_pid(target_file, pid=None):
    dirname = os.path.dirname(target_file)
    basename = os.path.basename(target_file)
    try:
        os.makedirs(dirname)
    except Exception:
        pass
    name, ext = os.path.splitext(basename)
    if pid is None:
        pid = os.getpid()
    return os.path.join(dirname, '%s.%s%s' % (name, pid, ext))