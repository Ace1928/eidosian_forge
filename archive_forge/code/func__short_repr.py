import inspect
import sys
def _short_repr(v):
    v = repr(v)
    if len(v) > 12:
        v = v[:8] + '...' + v[-4:]
    return v