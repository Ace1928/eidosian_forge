import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
@classmethod
def __subclasshook__(cls, subclass):
    if cls is PathLike:
        return _check_methods(subclass, '__fspath__')
    return NotImplemented