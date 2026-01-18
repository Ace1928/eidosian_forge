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
def _end_ns(self, prefix):
    return self.target.end_ns(prefix or '')