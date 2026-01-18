from __future__ import annotations
from typing import Any
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range
def _print_For(self, expr):
    target = self._print(expr.target)
    if isinstance(expr.iterable, Range):
        start, stop, step = expr.iterable.args
    else:
        raise NotImplementedError('Only iterable currently supported is Range')
    body = self._print(expr.body)
    return 'for({target} in seq(from={start}, to={stop}, by={step}){{\n{body}\n}}'.format(target=target, start=start, stop=stop - 1, step=step, body=body)