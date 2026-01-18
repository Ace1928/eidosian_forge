import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
class DynamicDimConstraintPrinter(StrPrinter):
    """
    Printer for dynamic dim constraints.
    - Instead of t.size()[d] it prints dynamic_dim(t, d)
    - Instead of Eq(_, _), Mod(_, _), etc. it prints _ == _, _ % _, etc.

    We use this to suggest code for specifying dynamic dim constraints.
    """

    def __init__(self, symbol_to_source, source_name_to_debug_name):
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_name_to_debug_name = source_name_to_debug_name

    def print_source(self, source) -> str:
        if self.source_name_to_debug_name:
            return source.name()
        return f'dynamic_dim({source.base.name()}, {source.idx})'

    def _print_Symbol(self, expr) -> str:
        assert isinstance(expr, sympy.Symbol), str(type(expr))
        return self.print_source(self.symbol_to_source[expr][0])

    def _print_Relational(self, expr):
        return '{} {} {}'.format(self.parenthesize(expr.lhs, precedence(expr)), expr.rel_op, self.parenthesize(expr.rhs, precedence(expr)))