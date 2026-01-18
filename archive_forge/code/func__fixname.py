import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def _fixname(self, key):
    try:
        name = self._names[key]
    except KeyError:
        name = key
        if '}' in name:
            name = '{' + name
        self._names[key] = name
    return name