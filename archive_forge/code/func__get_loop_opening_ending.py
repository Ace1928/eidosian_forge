from __future__ import annotations
from typing import Any
from sympy.core import S, Rational, Float, Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
def _get_loop_opening_ending(self, indices):
    open_lines = []
    close_lines = []
    loopstart = 'for %(var)s in %(start)s..%(end)s {'
    for i in indices:
        open_lines.append(loopstart % {'var': self._print(i), 'start': self._print(i.lower), 'end': self._print(i.upper + 1)})
        close_lines.append('}')
    return (open_lines, close_lines)