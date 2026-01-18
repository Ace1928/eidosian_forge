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
class _FunctionCompiler(object):

    def __init__(self, py_func, targetdescr, targetoptions, locals, pipeline_class):
        self.py_func = py_func
        self.targetdescr = targetdescr
        self.targetoptions = targetoptions
        self.locals = locals
        self.pysig = utils.pysignature(self.py_func)
        self.pipeline_class = pipeline_class
        self._failed_cache = {}

    def fold_argument_types(self, args, kws):
        """
        Given positional and named argument types, fold keyword arguments
        and resolve defaults by inserting types.Omitted() instances.

        A (pysig, argument types) tuple is returned.
        """

        def normal_handler(index, param, value):
            return value

        def default_handler(index, param, default):
            return types.Omitted(default)

        def stararg_handler(index, param, values):
            return types.StarArgTuple(values)
        args = fold_arguments(self.pysig, args, kws, normal_handler, default_handler, stararg_handler)
        return (self.pysig, args)

    def compile(self, args, return_type):
        status, retval = self._compile_cached(args, return_type)
        if status:
            return retval
        else:
            raise retval

    def _compile_cached(self, args, return_type):
        key = (tuple(args), return_type)
        try:
            return (False, self._failed_cache[key])
        except KeyError:
            pass
        try:
            retval = self._compile_core(args, return_type)
        except errors.TypingError as e:
            self._failed_cache[key] = e
            return (False, e)
        else:
            return (True, retval)

    def _compile_core(self, args, return_type):
        flags = compiler.Flags()
        self.targetdescr.options.parse_as_flags(flags, self.targetoptions)
        flags = self._customize_flags(flags)
        impl = self._get_implementation(args, {})
        cres = compiler.compile_extra(self.targetdescr.typing_context, self.targetdescr.target_context, impl, args=args, return_type=return_type, flags=flags, locals=self.locals, pipeline_class=self.pipeline_class)
        if cres.typing_error is not None and (not flags.enable_pyobject):
            raise cres.typing_error
        return cres

    def get_globals_for_reduction(self):
        return serialize._get_function_globals_for_reduction(self.py_func)

    def _get_implementation(self, args, kws):
        return self.py_func

    def _customize_flags(self, flags):
        return flags