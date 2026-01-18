import os
import re
import abc
import csv
import sys
import email
import pathlib
import zipfile
import operator
import textwrap
import warnings
import functools
import itertools
import posixpath
import collections
from . import _adapters, _meta
from ._collections import FreezableDefaultDict, Pair
from ._functools import method_cache, pass_none
from ._itertools import always_iterable, unique_everseen
from ._meta import PackageMetadata, SimplePath
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import List, Mapping, Optional, Union
class DeprecatedList(list):
    """
    Allow an otherwise immutable object to implement mutability
    for compatibility.

    >>> recwarn = getfixture('recwarn')
    >>> dl = DeprecatedList(range(3))
    >>> dl[0] = 1
    >>> dl.append(3)
    >>> del dl[3]
    >>> dl.reverse()
    >>> dl.sort()
    >>> dl.extend([4])
    >>> dl.pop(-1)
    4
    >>> dl.remove(1)
    >>> dl += [5]
    >>> dl + [6]
    [1, 2, 5, 6]
    >>> dl + (6,)
    [1, 2, 5, 6]
    >>> dl.insert(0, 0)
    >>> dl
    [0, 1, 2, 5]
    >>> dl == [0, 1, 2, 5]
    True
    >>> dl == (0, 1, 2, 5)
    True
    >>> len(recwarn)
    1
    """
    __slots__ = ()
    _warn = functools.partial(warnings.warn, 'EntryPoints list interface is deprecated. Cast to list if needed.', DeprecationWarning, stacklevel=2)

    def _wrap_deprecated_method(method_name: str):

        def wrapped(self, *args, **kwargs):
            self._warn()
            return getattr(super(), method_name)(*args, **kwargs)
        return (method_name, wrapped)
    locals().update(map(_wrap_deprecated_method, '__setitem__ __delitem__ append reverse extend pop remove __iadd__ insert sort'.split()))

    def __add__(self, other):
        if not isinstance(other, tuple):
            self._warn()
            other = tuple(other)
        return self.__class__(tuple(self) + other)

    def __eq__(self, other):
        if not isinstance(other, tuple):
            self._warn()
            other = tuple(other)
        return tuple(self).__eq__(other)