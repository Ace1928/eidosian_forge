from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
class ConcreteTemplate(FunctionTemplate):
    """
    Defines attributes "cases" as a list of signature to match against the
    given input types.
    """

    def apply(self, args, kws):
        cases = getattr(self, 'cases')
        return self._select(cases, args, kws)

    def get_template_info(self):
        import operator
        name = getattr(self.key, '__name__', 'unknown')
        op_func = getattr(operator, name, None)
        kind = 'Type restricted function'
        if op_func is not None:
            if self.key is op_func:
                kind = 'operator overload'
        info = {'kind': kind, 'name': name, 'sig': 'unknown', 'filename': 'unknown', 'lines': ('unknown', 'unknown'), 'docstring': 'unknown'}
        return info