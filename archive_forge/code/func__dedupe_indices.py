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
def _dedupe_indices(new, exclude):
    """
        exclude: set
        new: TensExpr

        If ``new`` has any dummy indices that are in ``exclude``, return a version
        of new with those indices replaced. If no replacements are needed,
        return None

        """
    exclude = set(exclude)
    dums_new = set(get_dummy_indices(new))
    free_new = set(get_free_indices(new))
    conflicts = dums_new.intersection(exclude)
    if len(conflicts) == 0:
        return None
    '\n        ``exclude_for_gen`` is to be passed to ``_IndexStructure._get_generator_for_dummy_indices()``.\n        Since the latter does not use the index position for anything, we just\n        set it as ``None`` here.\n        '
    exclude.update(dums_new)
    exclude.update(free_new)
    exclude_for_gen = [(i, None) for i in exclude]
    gen = _IndexStructure._get_generator_for_dummy_indices(exclude_for_gen)
    repl = {}
    for d in conflicts:
        if -d in repl.keys():
            continue
        newname = gen(d.tensor_index_type)
        new_d = d.func(newname, *d.args[1:])
        repl[d] = new_d
        repl[-d] = -new_d
    if len(repl) == 0:
        return None
    new_renamed = new._replace_indices(repl)
    return new_renamed