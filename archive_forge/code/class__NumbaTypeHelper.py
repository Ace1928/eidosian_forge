from llvmlite import ir
from numba.core import types, cgutils
from numba.core.pythonapi import box, unbox, reflect, NativeValue
from numba.core.errors import NumbaNotImplementedError, TypingError
from numba.core.typing.typeof import typeof, Purpose
from numba.cpython import setobj, listobj
from numba.np import numpy_support
from contextlib import contextmanager, ExitStack
class _NumbaTypeHelper(object):
    """A helper for acquiring `numba.typeof` for type checking.

    Usage
    -----

        # `c` is the boxing context.
        with _NumbaTypeHelper(c) as nth:
            # This contextmanager maintains the lifetime of the `numba.typeof`
            # function.
            the_numba_type = nth.typeof(some_object)
            # Do work on the type object
            do_checks(the_numba_type)
            # Cleanup
            c.pyapi.decref(the_numba_type)
        # At this point *nth* should not be used.
    """

    def __init__(self, c):
        self.c = c

    def __enter__(self):
        c = self.c
        numba_name = c.context.insert_const_string(c.builder.module, 'numba')
        numba_mod = c.pyapi.import_module_noblock(numba_name)
        typeof_fn = c.pyapi.object_getattr_string(numba_mod, 'typeof')
        self.typeof_fn = typeof_fn
        c.pyapi.decref(numba_mod)
        return self

    def __exit__(self, *args, **kwargs):
        c = self.c
        c.pyapi.decref(self.typeof_fn)

    def typeof(self, obj):
        res = self.c.pyapi.call_function_objargs(self.typeof_fn, [obj])
        return res