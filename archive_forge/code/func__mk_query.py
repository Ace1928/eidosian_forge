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
def _mk_query(name):
    original = getattr(_os, name)

    def wrap(self, *args, **kw):
        retval = original(*args, **kw)
        if self._active:
            return self._remap_output(name, retval)
        return retval
    return wrap