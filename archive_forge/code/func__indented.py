from __future__ import annotations
from typing import Any
from collections import defaultdict
from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
def _indented(self, printer, k, v, *args, **kwargs):
    il = printer._context['indent_level']

    def _print(arg):
        if isinstance(arg, Token):
            return printer._print(arg, *args, joiner=self._joiner(k, il), **kwargs)
        else:
            return printer._print(arg, *args, **kwargs)
    if isinstance(v, Tuple):
        joined = self._joiner(k, il).join([_print(arg) for arg in v.args])
        if k in self.indented_args:
            return '(\n' + ' ' * il + joined + ',\n' + ' ' * (il - 4) + ')'
        else:
            return ('({0},)' if len(v.args) == 1 else '({0})').format(joined)
    else:
        return _print(v)