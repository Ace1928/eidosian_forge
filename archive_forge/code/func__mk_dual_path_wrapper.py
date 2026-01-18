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
def _mk_dual_path_wrapper(name):
    original = getattr(_os, name)

    def wrap(self, src, dst, *args, **kw):
        if self._active:
            src, dst = self._remap_pair(name, src, dst, *args, **kw)
        return original(src, dst, *args, **kw)
    return wrap