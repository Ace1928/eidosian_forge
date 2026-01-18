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
class DelAttr(Stmt):

    def __init__(self, target, attr, loc):
        assert isinstance(target, Var)
        assert isinstance(attr, str)
        assert isinstance(loc, Loc)
        self.target = target
        self.attr = attr
        self.loc = loc

    def __repr__(self):
        return 'del (%s).%s' % (self.target, self.attr)