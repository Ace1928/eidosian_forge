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
class requires:
    """ Decorator for registering requirements on print methods. """

    def __init__(self, **kwargs):
        self._req = kwargs

    def __call__(self, method):

        def _method_wrapper(self_, *args, **kwargs):
            for k, v in self._req.items():
                getattr(self_, k).update(v)
            return method(self_, *args, **kwargs)
        return wraps(method)(_method_wrapper)