import collections.abc
import contextlib
import contextvars
from .._utils import set_module
from .umath import (
from . import umath
@set_module('numpy')
@contextlib.contextmanager
def _no_nep50_warning():
    """
    Context manager to disable NEP 50 warnings.  This context manager is
    only relevant if the NEP 50 warnings are enabled globally (which is not
    thread/context safe).

    This warning context manager itself is fully safe, however.
    """
    token = NO_NEP50_WARNING.set(True)
    try:
        yield
    finally:
        NO_NEP50_WARNING.reset(token)