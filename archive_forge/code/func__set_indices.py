from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
def _set_indices(self, *indices, is_canon_bp=False, **kw_args):
    if len(indices) != self.ext_rank:
        raise ValueError('indices length mismatch')
    args = list(self.args)[:]
    pos = 0
    for i, arg in enumerate(args):
        if not isinstance(arg, TensExpr):
            continue
        assert isinstance(arg, Tensor)
        ext_rank = arg.ext_rank
        args[i] = arg._set_indices(*indices[pos:pos + ext_rank])
        pos += ext_rank
    return TensMul(*args, is_canon_bp=is_canon_bp).doit()