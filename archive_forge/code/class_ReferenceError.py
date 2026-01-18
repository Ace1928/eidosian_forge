import weakref
import gc
from contextlib import contextmanager
from platform import python_implementation
class ReferenceError(AssertionError):
    pass