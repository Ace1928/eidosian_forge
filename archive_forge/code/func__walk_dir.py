import os
import sys
import importlib.util
import py_compile
import struct
import filecmp
from functools import partial
from pathlib import Path
def _walk_dir(dir, maxlevels, quiet=0):
    if quiet < 2 and isinstance(dir, os.PathLike):
        dir = os.fspath(dir)
    if not quiet:
        print('Listing {!r}...'.format(dir))
    try:
        names = os.listdir(dir)
    except OSError:
        if quiet < 2:
            print("Can't list {!r}".format(dir))
        names = []
    names.sort()
    for name in names:
        if name == '__pycache__':
            continue
        fullname = os.path.join(dir, name)
        if not os.path.isdir(fullname):
            yield fullname
        elif maxlevels > 0 and name != os.curdir and (name != os.pardir) and os.path.isdir(fullname) and (not os.path.islink(fullname)):
            yield from _walk_dir(fullname, maxlevels=maxlevels - 1, quiet=quiet)