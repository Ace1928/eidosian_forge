from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
@total_ordering
class SlotEqualityCheckMixin(object):
    __slots__ = tuple()

    def __eq__(self, other):
        if type(self) is type(other):
            for name in self.__slots__:
                if getattr(self, name) != getattr(other, name):
                    return False
            else:
                return True
        return False

    def __le__(self, other):
        return str(self) <= str(other)

    def __hash__(self):
        return id(self)