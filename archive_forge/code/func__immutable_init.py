import contextvars
import inspect
def _immutable_init(f):

    def nf(*args, **kwargs):
        previous = _in__init__.set(args[0])
        try:
            f(*args, **kwargs)
        finally:
            _in__init__.reset(previous)
    nf.__signature__ = inspect.signature(f)
    return nf