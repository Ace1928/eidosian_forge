import sys
def _reraise(cls, val, tb):
    __tracebackhide__ = True
    assert hasattr(val, '__traceback__')
    raise cls.with_traceback(val, tb)