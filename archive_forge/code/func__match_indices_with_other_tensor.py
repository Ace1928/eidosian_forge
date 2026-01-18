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
def _match_indices_with_other_tensor(array, free_ind1, free_ind2, replacement_dict):
    from .array import permutedims
    index_types1 = [i.tensor_index_type for i in free_ind1]
    pos2up = []
    pos2down = []
    free2remaining = free_ind2[:]
    for pos1, index1 in enumerate(free_ind1):
        if index1 in free2remaining:
            pos2 = free2remaining.index(index1)
            free2remaining[pos2] = None
            continue
        if -index1 in free2remaining:
            pos2 = free2remaining.index(-index1)
            free2remaining[pos2] = None
            free_ind2[pos2] = index1
            if index1.is_up:
                pos2up.append(pos2)
            else:
                pos2down.append(pos2)
        else:
            index2 = free2remaining[pos1]
            if index2 is None:
                raise ValueError('incompatible indices: %s and %s' % (free_ind1, free_ind2))
            free2remaining[pos1] = None
            free_ind2[pos1] = index1
            if index1.is_up ^ index2.is_up:
                if index1.is_up:
                    pos2up.append(pos1)
                else:
                    pos2down.append(pos1)
    if len(set(free_ind1) & set(free_ind2)) < len(free_ind1):
        raise ValueError('incompatible indices: %s and %s' % (free_ind1, free_ind2))
    for pos in pos2up:
        index_type_pos = index_types1[pos]
        if index_type_pos not in replacement_dict:
            raise ValueError('No metric provided to lower index')
        metric = replacement_dict[index_type_pos]
        metric_inverse = _TensorDataLazyEvaluator.inverse_matrix(metric)
        array = TensExpr._contract_and_permute_with_metric(metric_inverse, array, pos, len(free_ind1))
    for pos in pos2down:
        index_type_pos = index_types1[pos]
        if index_type_pos not in replacement_dict:
            raise ValueError('No metric provided to lower index')
        metric = replacement_dict[index_type_pos]
        array = TensExpr._contract_and_permute_with_metric(metric, array, pos, len(free_ind1))
    if free_ind1:
        permutation = TensExpr._get_indices_permutation(free_ind2, free_ind1)
        array = permutedims(array, permutation)
    if hasattr(array, 'rank') and array.rank() == 0:
        array = array[()]
    return (free_ind2, array)