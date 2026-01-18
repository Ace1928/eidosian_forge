from __future__ import annotations
import atexit
from contextlib import contextmanager
import fnmatch
import importlib.util
import io
import os
from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import isabs
from os.path import isdir
from os.path import isfile
from os.path import islink
from os.path import normpath
import posixpath
from stat import S_ISDIR
from stat import S_ISLNK
from stat import S_ISREG
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
from typing import overload
from typing import TYPE_CHECKING
import uuid
import warnings
from . import error
def _getbyspec(self, spec: str) -> list[str]:
    """See new for what 'spec' can be."""
    res = []
    parts = self.strpath.split(self.sep)
    args = filter(None, spec.split(','))
    for name in args:
        if name == 'drive':
            res.append(parts[0])
        elif name == 'dirname':
            res.append(self.sep.join(parts[:-1]))
        else:
            basename = parts[-1]
            if name == 'basename':
                res.append(basename)
            else:
                i = basename.rfind('.')
                if i == -1:
                    purebasename, ext = (basename, '')
                else:
                    purebasename, ext = (basename[:i], basename[i:])
                if name == 'purebasename':
                    res.append(purebasename)
                elif name == 'ext':
                    res.append(ext)
                else:
                    raise ValueError('invalid part specification %r' % name)
    return res