import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def _make_str_method(*args, **kwargs):
    """
    Generate a ``__str__`` method for a `.Transform` subclass.

    After ::

        class T:
            __str__ = _make_str_method("attr", key="other")

    ``str(T(...))`` will be

    .. code-block:: text

        {type(T).__name__}(
            {self.attr},
            key={self.other})
    """
    indent = functools.partial(textwrap.indent, prefix=' ' * 4)

    def strrepr(x):
        return repr(x) if isinstance(x, str) else str(x)
    return lambda self: type(self).__name__ + '(' + ','.join([*(indent('\n' + strrepr(getattr(self, arg))) for arg in args), *(indent('\n' + k + '=' + strrepr(getattr(self, arg))) for k, arg in kwargs.items())]) + ')'