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
def _ok(self, path):
    active = self._active
    try:
        self._active = False
        realpath = os.path.normcase(os.path.realpath(path))
        return self._exempted(realpath) or realpath == self._sandbox or realpath.startswith(self._prefix)
    finally:
        self._active = active