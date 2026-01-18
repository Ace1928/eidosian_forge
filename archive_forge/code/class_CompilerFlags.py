import sys
from enum import IntFlag
from _pydevd_frame_eval.vendored import bytecode as _bytecode
class CompilerFlags(IntFlag):
    """Possible values of the co_flags attribute of Code object.

    Note: We do not rely on inspect values here as some of them are missing and
    furthermore would be version dependent.

    """
    OPTIMIZED = 1
    NEWLOCALS = 2
    VARARGS = 4
    VARKEYWORDS = 8
    NESTED = 16
    GENERATOR = 32
    NOFREE = 64
    COROUTINE = 128
    ITERABLE_COROUTINE = 256
    ASYNC_GENERATOR = 512
    if sys.version_info < (3, 9):
        FUTURE_GENERATOR_STOP = 524288
        if sys.version_info > (3, 6):
            FUTURE_ANNOTATIONS = 1048576
    else:
        FUTURE_GENERATOR_STOP = 8388608
        FUTURE_ANNOTATIONS = 16777216