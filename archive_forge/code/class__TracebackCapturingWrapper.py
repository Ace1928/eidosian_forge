import ast
from dataclasses import dataclass
import operator as op
from ._multiprocessing_helpers import mp
class _TracebackCapturingWrapper:
    """Protect function call and return error with traceback."""

    def __init__(self, func):
        self.func = func

    def __call__(self, **kwargs):
        try:
            return self.func(**kwargs)
        except BaseException as e:
            return _ExceptionWithTraceback(e)