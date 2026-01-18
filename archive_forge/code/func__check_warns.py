import contextlib
import itertools
import re
import typing
from enum import Enum
from typing import Callable
import sympy
from sympy import Add, Implies, sqrt
from sympy.core import Mul, Pow
from sympy.core import (S, pi, symbols, Function, Rational, Integer,
from sympy.functions import Piecewise, exp, sin, cos
from sympy.printing.smtlib import smtlib_code
from sympy.testing.pytest import raises, Failed
@contextlib.contextmanager
def _check_warns(expected: typing.Iterable[_W]):
    warns: typing.List[str] = []
    log_warn = warns.append
    yield log_warn
    errors = []
    for i, (w, e) in enumerate(itertools.zip_longest(warns, expected)):
        if not e:
            errors += [f'[{i}] Received unexpected warning `{w}`.']
        elif not w:
            errors += [f'[{i}] Did not receive expected warning `{e.name}`.']
        elif not e.value.match(w):
            errors += [f'[{i}] Warning `{w}` does not match expected {e.name}.']
    if errors:
        raise Failed('\n'.join(errors))