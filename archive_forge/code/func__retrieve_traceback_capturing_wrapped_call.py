import ast
from dataclasses import dataclass
import operator as op
from ._multiprocessing_helpers import mp
def _retrieve_traceback_capturing_wrapped_call(out):
    if isinstance(out, _ExceptionWithTraceback):
        rebuild, args = out.__reduce__()
        out = rebuild(*args)
    if isinstance(out, BaseException):
        raise out
    return out