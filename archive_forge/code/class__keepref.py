import sys
from ctypes.util import find_library
from . import constants
from .ffi import ffi
from .surfaces import (  # noqa isort:skip
from .patterns import (  # noqa isort:skip
from .fonts import (  # noqa isort:skip
from .context import Context  # noqa isort:skip
from .matrix import Matrix  # noqa isort:skip
from .constants import *  # noqa isort:skip
class _keepref(object):
    """Function wrapper that keeps a reference to another object."""

    def __init__(self, ref, func):
        self.ref = ref
        self.func = func

    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)