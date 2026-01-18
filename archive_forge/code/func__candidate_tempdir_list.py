import functools as _functools
import warnings as _warnings
import io as _io
import os as _os
import shutil as _shutil
import stat as _stat
import errno as _errno
from random import Random as _Random
import sys as _sys
import types as _types
import weakref as _weakref
import _thread
def _candidate_tempdir_list():
    """Generate a list of candidate temporary directories which
    _get_default_tempdir will try."""
    dirlist = []
    for envname in ('TMPDIR', 'TEMP', 'TMP'):
        dirname = _os.getenv(envname)
        if dirname:
            dirlist.append(dirname)
    if _os.name == 'nt':
        dirlist.extend([_os.path.expanduser('~\\AppData\\Local\\Temp'), _os.path.expandvars('%SYSTEMROOT%\\Temp'), 'c:\\temp', 'c:\\tmp', '\\temp', '\\tmp'])
    else:
        dirlist.extend(['/tmp', '/var/tmp', '/usr/tmp'])
    try:
        dirlist.append(_os.getcwd())
    except (AttributeError, OSError):
        dirlist.append(_os.curdir)
    return dirlist