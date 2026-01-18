import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def _make_invoke_excepthook():
    old_excepthook = excepthook
    old_sys_excepthook = _sys.excepthook
    if old_excepthook is None:
        raise RuntimeError('threading.excepthook is None')
    if old_sys_excepthook is None:
        raise RuntimeError('sys.excepthook is None')
    sys_exc_info = _sys.exc_info
    local_print = print
    local_sys = _sys

    def invoke_excepthook(thread):
        global excepthook
        try:
            hook = excepthook
            if hook is None:
                hook = old_excepthook
            args = ExceptHookArgs([*sys_exc_info(), thread])
            hook(args)
        except Exception as exc:
            exc.__suppress_context__ = True
            del exc
            if local_sys is not None and local_sys.stderr is not None:
                stderr = local_sys.stderr
            else:
                stderr = thread._stderr
            local_print('Exception in threading.excepthook:', file=stderr, flush=True)
            if local_sys is not None and local_sys.excepthook is not None:
                sys_excepthook = local_sys.excepthook
            else:
                sys_excepthook = old_sys_excepthook
            sys_excepthook(*sys_exc_info())
        finally:
            args = None
    return invoke_excepthook