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
def _handle_single(self, factory, insert, *args):
    elem = factory(*args)
    if insert:
        self._flush()
        self._last = elem
        if self._elem:
            self._elem[-1].append(elem)
        self._tail = 1
    return elem