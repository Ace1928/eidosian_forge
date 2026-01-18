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
def _remap_pair(self, operation, src, dst, *args, **kw):
    """Called for path pairs like rename, link, and symlink operations"""
    if not self._ok(src) or not self._ok(dst):
        self._violation(operation, src, dst, *args, **kw)
    return (src, dst)