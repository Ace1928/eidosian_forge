import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
def _exempted(self, filepath):
    start_matches = (filepath.startswith(exception) for exception in self._exceptions)
    pattern_matches = (re.match(pattern, filepath) for pattern in self._exception_patterns)
    candidates = itertools.chain(start_matches, pattern_matches)
    return any(candidates)