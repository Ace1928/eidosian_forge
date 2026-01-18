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
def _match_indices_ignoring_order(self, expr, repl_dict=None, old=False):
    """
        Helper method for matches. Checks if the indices of self and expr
        match disregarding index ordering.
        """
    if repl_dict is None:
        repl_dict = {}
    else:
        repl_dict = repl_dict.copy()

    def siftkey(ind):
        if isinstance(ind, WildTensorIndex):
            if ind.ignore_updown:
                return 'wild, updown'
            else:
                return 'wild'
        else:
            return 'nonwild'
    indices_sifted = sift(self.indices, siftkey)
    matched_indices = []
    expr_indices_remaining = expr.get_indices()
    for ind in indices_sifted['nonwild']:
        matched_this_ind = False
        for e_ind in expr_indices_remaining:
            if e_ind in matched_indices:
                continue
            m = ind.matches(e_ind)
            if m is not None:
                matched_this_ind = True
                repl_dict.update(m)
                matched_indices.append(e_ind)
                break
        if not matched_this_ind:
            return None
    expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]
    for ind in indices_sifted['wild']:
        matched_this_ind = False
        for e_ind in expr_indices_remaining:
            m = ind.matches(e_ind)
            if m is not None:
                if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                    return None
                matched_this_ind = True
                repl_dict.update(m)
                matched_indices.append(e_ind)
                break
        if not matched_this_ind:
            return None
    expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]
    for ind in indices_sifted['wild, updown']:
        matched_this_ind = False
        for e_ind in expr_indices_remaining:
            m = ind.matches(e_ind)
            if m is not None:
                if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                    return None
                matched_this_ind = True
                repl_dict.update(m)
                matched_indices.append(e_ind)
                break
        if not matched_this_ind:
            return None
    if len(matched_indices) < len(self.indices):
        return None
    else:
        return repl_dict