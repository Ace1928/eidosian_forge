import collections
from contextlib import contextmanager, ExitStack
import functools
from llvmlite import ir
from numba.core import utils, types, config, debuginfo
import numba.core.datamodel
class ValueStructProxy(_StructProxy):
    """
    Create a StructProxy suitable for accessing regular values
    (e.g. LLVM values or alloca slots).
    """

    def _get_be_type(self, datamodel):
        return datamodel.get_value_type()

    def _cast_member_to_value(self, index, val):
        return val

    def _cast_member_from_value(self, index, val):
        return val