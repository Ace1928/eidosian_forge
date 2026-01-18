import collections.abc
import inspect
import os
import sys
import traceback
import types
def _format_coroutine(coro):
    assert iscoroutine(coro)

    def get_name(coro):
        if hasattr(coro, '__qualname__') and coro.__qualname__:
            coro_name = coro.__qualname__
        elif hasattr(coro, '__name__') and coro.__name__:
            coro_name = coro.__name__
        else:
            coro_name = f'<{type(coro).__name__} without __name__>'
        return f'{coro_name}()'

    def is_running(coro):
        try:
            return coro.cr_running
        except AttributeError:
            try:
                return coro.gi_running
            except AttributeError:
                return False
    coro_code = None
    if hasattr(coro, 'cr_code') and coro.cr_code:
        coro_code = coro.cr_code
    elif hasattr(coro, 'gi_code') and coro.gi_code:
        coro_code = coro.gi_code
    coro_name = get_name(coro)
    if not coro_code:
        if is_running(coro):
            return f'{coro_name} running'
        else:
            return coro_name
    coro_frame = None
    if hasattr(coro, 'gi_frame') and coro.gi_frame:
        coro_frame = coro.gi_frame
    elif hasattr(coro, 'cr_frame') and coro.cr_frame:
        coro_frame = coro.cr_frame
    filename = coro_code.co_filename or '<empty co_filename>'
    lineno = 0
    if coro_frame is not None:
        lineno = coro_frame.f_lineno
        coro_repr = f'{coro_name} running at {filename}:{lineno}'
    else:
        lineno = coro_code.co_firstlineno
        coro_repr = f'{coro_name} done, defined at {filename}:{lineno}'
    return coro_repr