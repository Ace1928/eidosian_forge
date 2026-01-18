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
class ObjModeLiftedWith(LiftedWith):

    def __init__(self, *args, **kwargs):
        self.output_types = kwargs.pop('output_types', None)
        super(LiftedWith, self).__init__(*args, **kwargs)
        if not self.flags.force_pyobject:
            raise ValueError('expecting `flags.force_pyobject`')
        if self.output_types is None:
            raise TypeError('`output_types` must be provided')
        self.flags.no_rewrites = True

    @property
    def _numba_type_(self):
        return types.ObjModeDispatcher(self)

    def get_call_template(self, args, kws):
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This enables the resolving of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        assert not kws
        self._legalize_arg_types(args)
        args = [types.ffi_forced_object] * len(args)
        if self._can_compile:
            self.compile(tuple(args))
        signatures = [typing.signature(self.output_types, *args)]
        pysig = None
        func_name = self.py_func.__name__
        name = 'CallTemplate({0})'.format(func_name)
        call_template = typing.make_concrete_template(name, key=func_name, signatures=signatures)
        return (call_template, pysig, args, kws)

    def _legalize_arg_types(self, args):
        for i, a in enumerate(args, start=1):
            if isinstance(a, types.List):
                msg = 'Does not support list type inputs into with-context for arg {}'
                raise errors.TypingError(msg.format(i))
            elif isinstance(a, types.Dispatcher):
                msg = 'Does not support function type inputs into with-context for arg {}'
                raise errors.TypingError(msg.format(i))

    @global_compiler_lock
    def compile(self, sig):
        args, _ = sigutils.normalize_signature(sig)
        sig = (types.ffi_forced_object,) * len(args)
        return super().compile(sig)