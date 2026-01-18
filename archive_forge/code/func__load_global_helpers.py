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
@utils.runonce
def _load_global_helpers():
    """
    Execute once to install special symbols into the LLVM symbol table.
    """
    ll.add_symbol('_Py_NoneStruct', id(None))
    for c_helpers in (_helperlib.c_helpers, _dynfunc.c_helpers):
        for py_name, c_address in c_helpers.items():
            c_name = 'numba_' + py_name
            ll.add_symbol(c_name, c_address)
    for obj in utils.builtins.__dict__.values():
        if isinstance(obj, type) and issubclass(obj, BaseException):
            ll.add_symbol('PyExc_%s' % obj.__name__, id(obj))