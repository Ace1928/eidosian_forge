from llvmlite.ir import Constant, IRBuilder
import llvmlite.ir
from numba.core import types, config, cgutils
class _ArgManager(object):
    """
    A utility class to handle argument unboxing and cleanup
    """

    def __init__(self, context, builder, api, env_manager, endblk, nargs):
        self.context = context
        self.builder = builder
        self.api = api
        self.env_manager = env_manager
        self.arg_count = 0
        self.cleanups = []
        self.nextblk = endblk

    def add_arg(self, obj, ty):
        """
        Unbox argument and emit code that handles any error during unboxing.
        Args are cleaned up in reverse order of the parameter list, and
        cleanup begins as soon as unboxing of any argument fails. E.g. failure
        on arg2 will result in control flow going through:

            arg2.err -> arg1.err -> arg0.err -> arg.end (returns)
        """
        native = self.api.to_native_value(ty, obj)
        with cgutils.if_unlikely(self.builder, native.is_error):
            self.builder.branch(self.nextblk)

        def cleanup_arg():
            self.api.reflect_native_value(ty, native.value, self.env_manager)
            if native.cleanup is not None:
                native.cleanup()
            if self.context.enable_nrt:
                self.context.nrt.decref(self.builder, ty, native.value)
        self.cleanups.append(cleanup_arg)
        cleanupblk = self.builder.append_basic_block('arg%d.err' % self.arg_count)
        with self.builder.goto_block(cleanupblk):
            cleanup_arg()
            self.builder.branch(self.nextblk)
        self.nextblk = cleanupblk
        self.arg_count += 1
        return native.value

    def emit_cleanup(self):
        """
        Emit the cleanup code after returning from the wrapped function.
        """
        for dtor in self.cleanups:
            dtor()