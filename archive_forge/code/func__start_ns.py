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
def _start_ns(self, prefix, uri):
    return self.target.start_ns(prefix or '', uri or '')