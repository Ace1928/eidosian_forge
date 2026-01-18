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
def TemporaryFile(mode='w+b', buffering=-1, encoding=None, newline=None, suffix=None, prefix=None, dir=None, *, errors=None):
    """Create and return a temporary file.
        Arguments:
        'prefix', 'suffix', 'dir' -- as for mkstemp.
        'mode' -- the mode argument to io.open (default "w+b").
        'buffering' -- the buffer size argument to io.open (default -1).
        'encoding' -- the encoding argument to io.open (default None)
        'newline' -- the newline argument to io.open (default None)
        'errors' -- the errors argument to io.open (default None)
        The file is created as mkstemp() would do it.

        Returns an object with a file-like interface.  The file has no
        name, and will cease to exist when it is closed.
        """
    global _O_TMPFILE_WORKS
    if 'b' not in mode:
        encoding = _io.text_encoding(encoding)
    prefix, suffix, dir, output_type = _sanitize_params(prefix, suffix, dir)
    flags = _bin_openflags
    if _O_TMPFILE_WORKS:
        fd = None

        def opener(*args):
            nonlocal fd
            flags2 = (flags | _os.O_TMPFILE) & ~_os.O_CREAT
            fd = _os.open(dir, flags2, 384)
            return fd
        try:
            file = _io.open(dir, mode, buffering=buffering, newline=newline, encoding=encoding, errors=errors, opener=opener)
            raw = getattr(file, 'buffer', file)
            raw = getattr(raw, 'raw', raw)
            raw.name = fd
            return file
        except IsADirectoryError:
            _O_TMPFILE_WORKS = False
        except OSError:
            pass
    fd = None

    def opener(*args):
        nonlocal fd
        fd, name = _mkstemp_inner(dir, prefix, suffix, flags, output_type)
        try:
            _os.unlink(name)
        except BaseException as e:
            _os.close(fd)
            raise
        return fd
    file = _io.open(dir, mode, buffering=buffering, newline=newline, encoding=encoding, errors=errors, opener=opener)
    raw = getattr(file, 'buffer', file)
    raw = getattr(raw, 'raw', raw)
    raw.name = fd
    return file