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
def _canonicalize_bool_expr_impl(expr: sympy.Expr):
    if isinstance(expr, (sympy.And, sympy.Or)):
        return type(expr)(*map(canonicalize_bool_expr, expr.args))
    opposite = {sympy.Gt: sympy.Lt, sympy.Ge: sympy.Le}
    if isinstance(expr, tuple(opposite.keys())):
        lhs = expr.rhs - expr.lhs
        t = opposite[type(expr)]
    else:
        assert isinstance(expr, (sympy.Lt, sympy.Le, sympy.Eq, sympy.Ne))
        lhs = expr.lhs - expr.rhs
        t = type(expr)
    rhs = 0
    if isinstance(lhs, sympy.Add):
        cts = []
        variables = []
        for term in lhs.args:
            if term.is_number:
                cts.append(term)
            else:
                variables.append(term)
        lhs = sympy.Add(*variables)
        rhs = -sympy.Add(*cts)
    return t(lhs, rhs)