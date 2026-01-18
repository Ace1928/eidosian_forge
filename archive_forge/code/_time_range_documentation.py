import functools
from cupy import cuda
from cupy_backends.cuda.api import runtime
Mark function calls with ranges using NVTX/rocTX. This object can be
    used either as a decorator or a context manager.

    When used as a decorator, the decorated function calls are marked as
    ranges:

    >>> from cupyx.profiler import time_range
    >>> @time_range()
    ... def function_to_profile():
    ...     pass

    When used as a context manager, it describes the enclosed block as a nested
    range:

    >>> from cupyx.profiler import time_range
    >>> with time_range('some range in green', color_id=0):
    ...    # do something you want to measure
    ...    pass

    The marked ranges are visible in the profiler (such as nvvp, nsys-ui, etc)
    timeline.

    Args:
        message (str): Name of a range. When used as a decorator, the default
            is ``func.__name__``.
        color_id: range color ID
        argb_color: range color in ARGB (e.g. 0xFF00FF00 for green)
        sync (bool): If ``True``, waits for completion of all outstanding
            processing on GPU before calling :func:`cupy.cuda.nvtx.RangePush()`
            or :func:`cupy.cuda.nvtx.RangePop()`

    .. seealso:: :func:`cupy.cuda.nvtx.RangePush`,
        :func:`cupy.cuda.nvtx.RangePop`
    