import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
class CppPrinter(ExprPrinter):

    def _print_Integer(self, expr):
        return f'{int(expr)}L'

    def _print_Where(self, expr):
        c = self.paren(self.doprint(expr.args[0]))
        p = self.paren(self.doprint(expr.args[1]))
        q = self.paren(self.doprint(expr.args[2]))
        return f'{c} ? {p} : {q}'

    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        if div != 1:
            div = self.paren(self.doprint(div))
            if expr.is_integer:
                x = f'c10::div_floor_integer({x}, {div})'
            else:
                x = f'c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))'
        mod = self.paren(self.doprint(mod))
        return f'static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod})'

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        if expr.is_integer:
            return f'c10::div_floor_integer({x}, {div})'
        return f'c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))'

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        r = f'std::floor({self._print(expr.args[0])})'
        return f'static_cast<{INDEX_TYPE}>({r})' if expr.is_integer else r

    def _print_Pow(self, expr):
        base, exp = expr.args
        base = self._print(base)
        if exp == 0.5 or exp == -0.5:
            return f'std::sqrt({base})' if exp == 0.5 else f'1.0/std::sqrt({base})'
        assert exp.is_integer
        exp = int(exp)
        if exp > 0:
            r = '*'.join([self.paren(base)] * exp)
        elif exp < 0:
            r = '1.0/' + self.paren('*'.join([self.paren(base)] * abs(exp)))
        else:
            r = '1.0'
        return f'static_cast<{INDEX_TYPE}>({r})' if expr.is_integer else r

    def _print_Rational(self, expr):
        if expr.q == 1:
            r = f'{expr.p}'
        else:
            r = f'{expr.p}.0/{expr.q}.0'
        return f'static_cast<{INDEX_TYPE}>({r})' if expr.is_integer else r

    def _print_ceiling(self, expr):
        assert len(expr.args) == 1
        r = f'std::ceil({self._print(expr.args[0])})'
        return f'static_cast<{INDEX_TYPE}>({r})' if expr.is_integer else r

    def _print_Min(self, expr):
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f'std::min({args[0]}, {args[1]})'
        else:
            il = '{' + ', '.join(args) + '}'
            return f'std::min({il})'

    def _print_Max(self, expr):
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f'std::max({args[0]}, {args[1]})'
        else:
            il = '{' + ', '.join(args) + '}'
            return f'std::max({il})'

    def _print_Abs(self, expr):
        assert len(expr.args) == 1
        return f'std::abs({self._print(expr.args[0])})'