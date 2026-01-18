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
class LiftedCode(serialize.ReduceMixin, _MemoMixin, _DispatcherBase):
    """
    Implementation of the hidden dispatcher objects used for lifted code
    (a lifted loop is really compiled as a separate function).
    """
    _fold_args = False
    can_cache = False

    def __init__(self, func_ir, typingctx, targetctx, flags, locals):
        self.func_ir = func_ir
        self.lifted_from = None
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.flags = flags
        self.locals = locals
        _DispatcherBase.__init__(self, self.func_ir.arg_count, self.func_ir.func_id.func, self.func_ir.func_id.pysig, can_fallback=True, exact_match_required=False)

    def _reduce_states(self):
        """
        Reduce the instance for pickling.  This will serialize
        the original function as well the compilation options and
        compiled signatures, but not the compiled code itself.

        NOTE: part of ReduceMixin protocol
        """
        return dict(uuid=self._uuid, func_ir=self.func_ir, flags=self.flags, locals=self.locals, extras=self._reduce_extras())

    def _reduce_extras(self):
        """
        NOTE: sub-class can override to add extra states
        """
        return {}

    @classmethod
    def _rebuild(cls, uuid, func_ir, flags, locals, extras):
        """
        Rebuild an Dispatcher instance after it was __reduce__'d.

        NOTE: part of ReduceMixin protocol
        """
        try:
            return cls._memo[uuid]
        except KeyError:
            pass
        from numba.core import registry
        typingctx = registry.cpu_target.typing_context
        targetctx = registry.cpu_target.target_context
        self = cls(func_ir, typingctx, targetctx, flags, locals, **extras)
        self._set_uuid(uuid)
        return self

    def get_source_location(self):
        """Return the starting line number of the loop.
        """
        return self.func_ir.loc.line

    def _pre_compile(self, args, return_type, flags):
        """Pre-compile actions
        """
        pass

    @abstractmethod
    def compile(self, sig):
        """Lifted code should implement a compilation method that will return
        a CompileResult.entry_point for the given signature."""
        pass

    def _get_dispatcher_for_current_target(self):
        return self