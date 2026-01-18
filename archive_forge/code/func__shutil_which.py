import functools
import os
import sys
import re
import shutil
import types
from .encoding import DEFAULT_ENCODING
import platform
def _shutil_which(cmd, mode=os.F_OK | os.X_OK, path=None):
    """Given a command, mode, and a PATH string, return the path which
    conforms to the given mode on the PATH, or None if there is no such
    file.

    `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
    of os.environ.get("PATH"), or can be overridden with a custom search
    path.
    
    This is a backport of shutil.which from Python 3.4
    """

    def _access_check(fn, mode):
        return os.path.exists(fn) and os.access(fn, mode) and (not os.path.isdir(fn))
    if os.path.dirname(cmd):
        if _access_check(cmd, mode):
            return cmd
        return None
    if path is None:
        path = os.environ.get('PATH', os.defpath)
    if not path:
        return None
    path = path.split(os.pathsep)
    if sys.platform == 'win32':
        if not os.curdir in path:
            path.insert(0, os.curdir)
        pathext = os.environ.get('PATHEXT', '').split(os.pathsep)
        if any((cmd.lower().endswith(ext.lower()) for ext in pathext)):
            files = [cmd]
        else:
            files = [cmd + ext for ext in pathext]
    else:
        files = [cmd]
    seen = set()
    for dir in path:
        normdir = os.path.normcase(dir)
        if not normdir in seen:
            seen.add(normdir)
            for thefile in files:
                name = os.path.join(dir, thefile)
                if _access_check(name, mode):
                    return name
    return None