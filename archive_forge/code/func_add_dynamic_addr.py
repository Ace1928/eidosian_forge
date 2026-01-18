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
def add_dynamic_addr(self, builder, intaddr, info):
    """
        Returns dynamic address as a void pointer `i8*`.

        Internally, a global variable is added to inform the lowerer about
        the usage of dynamic addresses.  Caching will be disabled.
        """
    assert self.allow_dynamic_globals, 'dyn globals disabled in this target'
    assert isinstance(intaddr, int), 'dyn addr not of int type'
    mod = builder.module
    llvoidptr = self.get_value_type(types.voidptr)
    addr = self.get_constant(types.uintp, intaddr).inttoptr(llvoidptr)
    symname = 'numba.dynamic.globals.{:x}'.format(intaddr)
    gv = cgutils.add_global_variable(mod, llvoidptr, symname)
    gv.linkage = 'linkonce'
    gv.initializer = addr
    return builder.load(gv)