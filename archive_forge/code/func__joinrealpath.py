import os
import sys
import stat
import genericpath
from genericpath import *
def _joinrealpath(path, rest, strict, seen):
    if isinstance(path, bytes):
        sep = b'/'
        curdir = b'.'
        pardir = b'..'
    else:
        sep = '/'
        curdir = '.'
        pardir = '..'
    if isabs(rest):
        rest = rest[1:]
        path = sep
    while rest:
        name, _, rest = rest.partition(sep)
        if not name or name == curdir:
            continue
        if name == pardir:
            if path:
                path, name = split(path)
                if name == pardir:
                    path = join(path, pardir, pardir)
            else:
                path = pardir
            continue
        newpath = join(path, name)
        try:
            st = os.lstat(newpath)
        except OSError:
            if strict:
                raise
            is_link = False
        else:
            is_link = stat.S_ISLNK(st.st_mode)
        if not is_link:
            path = newpath
            continue
        if newpath in seen:
            path = seen[newpath]
            if path is not None:
                continue
            if strict:
                os.stat(newpath)
            else:
                return (join(newpath, rest), False)
        seen[newpath] = None
        path, ok = _joinrealpath(path, os.readlink(newpath), strict, seen)
        if not ok:
            return (join(path, rest), False)
        seen[newpath] = path
    return (path, True)