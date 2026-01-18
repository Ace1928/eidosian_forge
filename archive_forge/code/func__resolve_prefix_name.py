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
def _resolve_prefix_name(self, prefixed_name):
    prefix, name = prefixed_name.split(':', 1)
    for uri, p in self._iter_namespaces(self._ns_stack):
        if p == prefix:
            return f'{{{uri}}}{name}'
    raise ValueError(f'Prefix {prefix} of QName "{prefixed_name}" is not declared in scope')