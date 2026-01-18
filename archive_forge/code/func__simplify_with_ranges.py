import functools
import itertools
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import bound_sympy
from .utils import sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
def _simplify_with_ranges(self, expr: Expr, var_ranges: VarRanges) -> Expr:
    """
        Simplify indexing expression with knowledge of the ranges of
        iteration variables.
        """
    expr = join_dimensions(self.simplify(expr))
    original_expr = expr

    def remove_zero_terms(base, divisor):
        """Symbols smaller than the divisor are zero"""
        for v in base.free_symbols:
            if v in var_ranges:
                rest = sympy.Wild('_rest', exclude=[v])
                m = base.match(v + rest)
                if m and v not in m[rest].free_symbols:
                    gcd = sympy.gcd(m[rest], divisor)
                    if gcd == divisor:
                        if self.statically_known_leq(var_ranges[v], divisor):
                            base = m[rest]
        return base

    def visit_indexing_div(base, divisor):
        return FloorDiv(remove_zero_terms(base, divisor), divisor)

    def visit_modular_indexing(base, divisor, modulus):
        base = remove_zero_terms(base, divisor)
        base_pos = True
        if isinstance(base, ModularIndexing):
            base_s = base.args[2] - 1
        elif not base.has(ModularIndexing):
            iter_ranges_zero = {k: 0 for k, v in var_ranges.items()}
            base_lowest = sympy_subs(base, iter_ranges_zero)
            if self.statically_known_leq(0, base_lowest):
                base_pos = True
            else:
                base_pos = False
            iter_ranges = {k: v - 1 for k, v in var_ranges.items()}
            base_s = sympy_subs(base, iter_ranges)
        else:
            base_s = base
        if self.statically_known_lt(base_s, modulus * divisor) and base_pos:
            return FloorDiv(base, divisor)
        return ModularIndexing(base, divisor, modulus)
    if expr.has(ModularIndexing):
        expr = expr.replace(ModularIndexing(sympy.Wild('base'), sympy.Wild('divisor'), sympy.Wild('modulus')), visit_modular_indexing)
    if expr.has(FloorDiv):
        expr = expr.replace(FloorDiv(sympy.Wild('base'), sympy.Wild('divisor')), visit_indexing_div)
    if expr != original_expr:
        return self._simplify_with_ranges(expr, var_ranges)
    return expr