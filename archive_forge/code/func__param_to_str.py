import itertools
import types
import unittest
from cupy.testing import _bundle
from cupy.testing import _pytest_impl
def _param_to_str(obj):
    if isinstance(obj, type):
        return obj.__name__
    elif hasattr(obj, '__name__') and isinstance(obj.__name__, str):
        return obj.__name__
    return repr(obj)