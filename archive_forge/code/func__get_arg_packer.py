from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def _get_arg_packer(self, argtypes):
    """
        Get an argument packer for the given argument types.
        """
    return self.context.get_arg_packer(argtypes)