from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
class ErrorModel(object):

    def __init__(self, call_conv):
        self.call_conv = call_conv

    def fp_zero_division(self, builder, exc_args=None, loc=None):
        if self.raise_on_fp_zero_division:
            self.call_conv.return_user_exc(builder, ZeroDivisionError, exc_args, loc)
            return True
        else:
            return False