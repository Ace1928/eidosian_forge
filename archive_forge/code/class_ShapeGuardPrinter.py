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
class ShapeGuardPrinter(StrPrinter):

    def __init__(self, symbol_to_source, source_ref, var_to_sources):
        super().__init__()
        self.symbol_to_source = symbol_to_source
        self.source_ref = source_ref
        self.var_to_sources = var_to_sources

    def _print_Not(self, expr):
        return 'not %s' % self.parenthesize(expr.args[0], PRECEDENCE['Not'])

    def _print_And(self, expr):
        return self.stringify(expr.args, ' and ', PRECEDENCE['And'])

    def _print_Or(self, expr):
        return self.stringify(expr.args, ' or ', PRECEDENCE['Or'])

    def _print_Symbol(self, expr) -> str:
        assert isinstance(expr, sympy.Symbol), str(type(expr))

        def repr_symbol_to_source():
            return repr({symbol: [s.name() for s in sources] for symbol, sources in self.symbol_to_source.items()})
        assert self.symbol_to_source.get(expr), f'{expr} (could be from {[s.name() for s in self.var_to_sources[expr]]}) not in {repr_symbol_to_source()}.  If this assert is failing, it could be due to the issue described in https://github.com/pytorch/pytorch/pull/90665'
        return self.source_ref(self.symbol_to_source[expr][0])