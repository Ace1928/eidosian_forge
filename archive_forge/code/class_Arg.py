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
class Arg(EqualityCheckMixin, AbstractRHS):

    def __init__(self, name, index, loc):
        assert isinstance(name, str)
        assert isinstance(index, int)
        assert isinstance(loc, Loc)
        self.name = name
        self.index = index
        self.loc = loc

    def __repr__(self):
        return 'arg(%d, name=%s)' % (self.index, self.name)

    def infer_constant(self):
        raise ConstantInferenceError('%s' % self, loc=self.loc)