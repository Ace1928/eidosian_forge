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
def _end(self, tag):
    return self.target.end(self._fixname(tag))