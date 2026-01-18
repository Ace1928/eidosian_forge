from collections import defaultdict
import copy
import sys
from itertools import permutations, takewhile
from contextlib import contextmanager
from functools import cached_property
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
import llvmlite.binding as ll
from numba.core import types, utils, datamodel, debuginfo, funcdesc, config, cgutils, imputils
from numba.core import event, errors, targetconfig
from numba import _dynfunc, _helperlib
from numba.core.compiler_lock import global_compiler_lock
from numba.core.pythonapi import PythonAPI
from numba.core.imputils import (user_function, user_generator,
from numba.cpython import builtins
class _wrap_missing_loc(object):

    def __init__(self, fn):
        self.func = fn

    def __call__(self):
        """Wrap function for missing ``loc`` keyword argument.
        Otherwise, return the original *fn*.
        """
        fn = self.func
        if not _has_loc(fn):

            def wrapper(*args, **kwargs):
                kwargs.pop('loc')
                return fn(*args, **kwargs)
            attrs = ('__name__', 'libs')
            for attr in attrs:
                try:
                    val = getattr(fn, attr)
                except AttributeError:
                    pass
                else:
                    setattr(wrapper, attr, val)
            return wrapper
        else:
            return fn

    def __repr__(self):
        return '<wrapped %s>' % self.func