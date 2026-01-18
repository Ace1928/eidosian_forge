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
class VarMap(object):

    def __init__(self):
        self._con = {}

    def define(self, name, var):
        if name in self._con:
            raise RedefinedError(name)
        else:
            self._con[name] = var

    def get(self, name):
        try:
            return self._con[name]
        except KeyError:
            raise NotDefinedError(name)

    def __contains__(self, name):
        return name in self._con

    def __len__(self):
        return len(self._con)

    def __repr__(self):
        return pprint.pformat(self._con)

    def __hash__(self):
        return hash(self.name)

    def __iter__(self):
        return self._con.iterkeys()

    def __eq__(self, other):
        if type(self) is type(other):
            return self._con.keys() == other._con.keys()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)