from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def _add_exception(self, exc, exc_args, locinfo):
    """
        Add a new user exception to this helper. Returns an integer that can be
        used to refer to the added exception in future.

        Parameters
        ----------
        exc :
            exception type
        exc_args : None or tuple
            exception args
        locinfo : tuple
            location information
        """
    exc_id = len(self.exceptions) + FIRST_USEREXC
    self.exceptions[exc_id] = (exc, exc_args, locinfo)
    return exc_id