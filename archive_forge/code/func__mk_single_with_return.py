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
def _mk_single_with_return(name):
    original = getattr(_os, name)

    def wrap(self, path, *args, **kw):
        if self._active:
            path = self._remap_input(name, path, *args, **kw)
            return self._remap_output(name, original(path, *args, **kw))
        return original(path, *args, **kw)
    return wrap