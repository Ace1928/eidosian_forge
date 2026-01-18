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
@_lru_cache
def _maybe_evaluate_static(self, expr: 'sympy.Expr', *, unbacked_only: bool=False, compute_hint: bool=False, expect_rational=True) -> 'Optional[sympy.Expr]':
    """
        Tries to evaluate expr without introducing guards

        If unbacked_only == True, then we only do substitutions on
        unbacked SymInts (leaving regular hinted integers alone).  This could
        result in an expression that still contains backed SymInts, which you
        could then potentially guard on.

        Use compute_hint == True if you are trying to compute a non-binding
        hint for the particular hint values of backed SymInts, e.g., if
        s0 happens to be 3 this run, compute_hint will subsitute s0 with 3.
        """
    expr = self.simplify(expr)
    if compute_hint:
        expr = expr.xreplace(self.var_to_val)
    expr = canonicalize_bool_expr(expr)
    symbols = list(expr.free_symbols)
    for s in symbols:
        if s in self.var_to_val:
            continue
        subst = {}
        for ra in self.deferred_runtime_asserts.get(s, ()):
            if compute_hint:
                e = canonicalize_bool_expr(ra.expr.xreplace(self.var_to_val))
            else:
                e = ra.expr
            subst[e] = sympy.true
            subst[canonicalize_bool_expr(sympy.Not(e))] = sympy.false
            if isinstance(e, sympy.Eq):
                subst[sympy.Le(e.lhs, e.rhs)] = sympy.true
                subst[sympy.Le(-e.lhs, -e.rhs)] = sympy.true
                subst[sympy.Lt(e.lhs, e.rhs)] = sympy.false
                subst[sympy.Lt(-e.lhs, -e.rhs)] = sympy.false
        expr = expr.subs(subst)
    new_shape_env = {}
    new_range_env = {}
    for idx, k in enumerate(symbols):
        if isinstance(self.var_to_val.get(k, None), SingletonInt):
            continue
        vr = self.var_to_range[k]
        if vr.lower < (-sys.maxsize - 1) // 2 or (unbacked_only and k in self.var_to_val):
            new_range_env[k] = vr
            continue
        s = sympy.Symbol(f'shape_{idx}', positive=True, integer=True)
        offset = vr.lower - 1
        new_shape_env[k] = s + offset
        new_range_env[s] = SymPyValueRangeAnalysis.add(vr, -offset)

    def replace(expr, repl):
        return expr.xreplace(repl)
    try:
        new_expr = replace(expr, new_shape_env)
    except RecursionError:
        log.warning('RecursionError in sympy.xreplace(%s, %s)', expr, new_shape_env)
        self.counter['sympy_recursion_error'] += 1
        return None
    floor_div_replace = {}
    for atom in new_expr.atoms(FloorDiv):
        floor_div_replace[atom] = sympy.floor(atom.args[0] / atom.args[1])
    new_expr = safe_expand(new_expr.xreplace(floor_div_replace))
    if new_expr.is_number:
        return new_expr
    out = bound_sympy(new_expr, new_range_env)
    if expect_rational:
        _assert_bound_is_rational(new_expr, out)
        if out.is_singleton():
            return out.lower
    return new_expr if unbacked_only else None