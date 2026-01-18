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
def _iter_namespaces(self, ns_stack, _reversed=reversed):
    for namespaces in _reversed(ns_stack):
        if namespaces:
            yield from namespaces