from __future__ import annotations
from typing import Any
from functools import wraps
from sympy.core import Add, Mul, Pow, S, sympify, Float
from sympy.core.basic import Basic
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Lambda
from sympy.core.mul import _keep_coeff
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import re
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def _doprint_loops(self, expr, assign_to=None):
    if self._settings.get('contract', True):
        from sympy.tensor import get_contraction_structure
        indices = self._get_expression_indices(expr, assign_to)
        dummies = get_contraction_structure(expr)
    else:
        indices = []
        dummies = {None: (expr,)}
    openloop, closeloop = self._get_loop_opening_ending(indices)
    if None in dummies:
        text = StrPrinter.doprint(self, Add(*dummies[None]))
    else:
        text = StrPrinter.doprint(self, 0)
    lhs_printed = self._print(assign_to)
    lines = []
    if text != lhs_printed:
        lines.extend(openloop)
        if assign_to is not None:
            text = self._get_statement('%s = %s' % (lhs_printed, text))
        lines.append(text)
        lines.extend(closeloop)
    for d in dummies:
        if isinstance(d, tuple):
            indices = self._sort_optimized(d, expr)
            openloop_d, closeloop_d = self._get_loop_opening_ending(indices)
            for term in dummies[d]:
                if term in dummies and (not [list(f.keys()) for f in dummies[term]] == [[None] for f in dummies[term]]):
                    raise NotImplementedError('FIXME: no support for contractions in factor yet')
                else:
                    if assign_to is None:
                        raise AssignmentError('need assignment variable for loops')
                    if term.has(assign_to):
                        raise ValueError('FIXME: lhs present in rhs,                                this is undefined in CodePrinter')
                    lines.extend(openloop)
                    lines.extend(openloop_d)
                    text = '%s = %s' % (lhs_printed, StrPrinter.doprint(self, assign_to + term))
                    lines.append(self._get_statement(text))
                    lines.extend(closeloop_d)
                    lines.extend(closeloop)
    return '\n'.join(lines)