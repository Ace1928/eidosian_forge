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
def _close_and_return_root(self):
    root = self._parser.close()
    self._parser = None
    return root