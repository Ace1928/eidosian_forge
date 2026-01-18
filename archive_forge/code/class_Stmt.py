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
class Stmt(Inst):
    """
    Base class for IR statements (instructions which can appear on their
    own in a Block).
    """
    is_terminator = False
    is_exit = False

    def list_vars(self):
        return self._rec_list_vars(self.__dict__)