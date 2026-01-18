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
def _find_definition(self):
    fn_name = None
    lines = self.get_lines()
    for x in reversed(lines[:self.line - 1]):
        if x.strip().startswith('def '):
            fn_name = x
            break
    return fn_name