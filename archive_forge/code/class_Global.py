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
class Global(EqualityCheckMixin, AbstractRHS):

    def __init__(self, name, value, loc):
        assert isinstance(loc, Loc)
        self.name = name
        self.value = value
        self.loc = loc

    def __str__(self):
        return 'global(%s: %s)' % (self.name, self.value)

    def infer_constant(self):
        return self.value

    def __deepcopy__(self, memo):
        return Global(self.name, self.value, copy.deepcopy(self.loc))