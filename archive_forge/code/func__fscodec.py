import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
def _fscodec():
    encoding = sys.getfilesystemencoding()
    errors = sys.getfilesystemencodeerrors()

    def fsencode(filename):
        """Encode filename (an os.PathLike, bytes, or str) to the filesystem
        encoding with 'surrogateescape' error handler, return bytes unchanged.
        On Windows, use 'strict' error handler if the file system encoding is
        'mbcs' (which is the default encoding).
        """
        filename = fspath(filename)
        if isinstance(filename, str):
            return filename.encode(encoding, errors)
        else:
            return filename

    def fsdecode(filename):
        """Decode filename (an os.PathLike, bytes, or str) from the filesystem
        encoding with 'surrogateescape' error handler, return str unchanged. On
        Windows, use 'strict' error handler if the file system encoding is
        'mbcs' (which is the default encoding).
        """
        filename = fspath(filename)
        if isinstance(filename, bytes):
            return filename.decode(encoding, errors)
        else:
            return filename
    return (fsencode, fsdecode)