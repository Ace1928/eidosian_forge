import collections
import functools
import sys
import types as pytypes
import uuid
import weakref
from contextlib import ExitStack
from abc import abstractmethod
from numba import _dispatcher
from numba.core import (
from numba.core.compiler_lock import global_compiler_lock
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import fold_arguments
from numba.core.typing.typeof import Purpose, typeof
from numba.core.bytecode import get_code_object
from numba.core.caching import NullCache, FunctionCache
from numba.core import entrypoints
from numba.core.retarget import BaseRetarget
import numba.core.event as ev
class _GeneratedFunctionCompiler(_FunctionCompiler):

    def __init__(self, py_func, targetdescr, targetoptions, locals, pipeline_class):
        super(_GeneratedFunctionCompiler, self).__init__(py_func, targetdescr, targetoptions, locals, pipeline_class)
        self.impls = set()

    def get_globals_for_reduction(self):
        return serialize._get_function_globals_for_reduction(self.py_func)

    def _get_implementation(self, args, kws):
        impl = self.py_func(*args, **kws)
        pysig = utils.pysignature(self.py_func)
        implsig = utils.pysignature(impl)
        ok = len(pysig.parameters) == len(implsig.parameters)
        if ok:
            for pyparam, implparam in zip(pysig.parameters.values(), implsig.parameters.values()):
                if pyparam.name != implparam.name or pyparam.kind != implparam.kind or (implparam.default is not implparam.empty and implparam.default != pyparam.default):
                    ok = False
        if not ok:
            raise TypeError("generated implementation %s should be compatible with signature '%s', but has signature '%s'" % (impl, pysig, implsig))
        self.impls.add(impl)
        return impl