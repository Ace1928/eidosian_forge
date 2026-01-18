import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
def _fspath(path):
    """Return the path representation of a path-like object.

    If str or bytes is passed in, it is returned unchanged. Otherwise the
    os.PathLike interface is used to get the path representation. If the
    path representation is not str or bytes, TypeError is raised. If the
    provided path is not str, bytes, or os.PathLike, TypeError is raised.
    """
    if isinstance(path, (str, bytes)):
        return path
    path_type = type(path)
    try:
        path_repr = path_type.__fspath__(path)
    except AttributeError:
        if hasattr(path_type, '__fspath__'):
            raise
        else:
            raise TypeError('expected str, bytes or os.PathLike object, not ' + path_type.__name__)
    if isinstance(path_repr, (str, bytes)):
        return path_repr
    else:
        raise TypeError('expected {}.__fspath__() to return str or bytes, not {}'.format(path_type.__name__, type(path_repr).__name__))