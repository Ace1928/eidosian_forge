from __future__ import annotations
import re
import typing
from itertools import product
from typing import Any, Callable
import sympy
from sympy import Mul, Add, Pow, log, exp, sqrt, cos, sin, tan, asin, acos, acot, asec, acsc, sinh, cosh, tanh, asinh, \
from sympy.core.sympify import sympify, _sympify
from sympy.functions.special.bessel import airybiprime
from sympy.functions.special.error_functions import li
from sympy.utilities.exceptions import sympy_deprecation_warning
def _convert_one_function(self, s, fm, args, bgn, end):
    if (fm, len(args)) in self.translations:
        key = (fm, len(args))
        x_args = self.translations[key]['args']
        d = {k: v for k, v in zip(x_args, args)}
    elif (fm, '*') in self.translations:
        key = (fm, '*')
        x_args = self.translations[key]['args']
        d = {}
        for i, x in enumerate(x_args):
            if x[0] == '*':
                d[x] = ','.join(args[i:])
                break
            d[x] = args[i]
    else:
        err = "'{f}' is out of the whitelist.".format(f=fm)
        raise ValueError(err)
    template = self.translations[key]['fs']
    pat = self.translations[key]['pat']
    scanned = ''
    cur = 0
    while True:
        m = pat.search(template)
        if m is None:
            scanned += template
            break
        x = m.group()
        xbgn = m.start()
        scanned += template[:xbgn] + d[x]
        cur = m.end()
        template = template[cur:]
    s = s[:bgn] + scanned + s[end:]
    return s