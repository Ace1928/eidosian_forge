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
def _remap_input(self, operation, path, *args, **kw):
    """Called for path inputs"""
    if operation in self.write_ops and (not self._ok(path)):
        self._violation(operation, os.path.realpath(path), *args, **kw)
    return path