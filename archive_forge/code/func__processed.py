import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
@cached_property
def _processed(self):
    """A cached property for lazily processing the data and returning it.

        See ``_process()`` for details.
        """
    return self._process()