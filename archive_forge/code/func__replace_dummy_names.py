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
@staticmethod
def _replace_dummy_names(indices, free, dum):
    dum.sort(key=lambda x: x[0])
    new_indices = list(indices)
    assert len(indices) == len(free) + 2 * len(dum)
    generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
    for ipos1, ipos2 in dum:
        typ1 = new_indices[ipos1].tensor_index_type
        indname = generate_dummy_name(typ1)
        new_indices[ipos1] = TensorIndex(indname, typ1, True)
        new_indices[ipos2] = TensorIndex(indname, typ1, False)
    return new_indices