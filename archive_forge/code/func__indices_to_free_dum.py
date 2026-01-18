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
def _indices_to_free_dum(args_indices):
    free2pos1 = {}
    free2pos2 = {}
    dummy_data = []
    indices = []
    pos2 = 0
    for pos1, arg_indices in enumerate(args_indices):
        for index_pos, index in enumerate(arg_indices):
            if not isinstance(index, TensorIndex):
                raise TypeError('expected TensorIndex')
            if -index in free2pos1:
                other_pos1 = free2pos1.pop(-index)
                other_pos2 = free2pos2.pop(-index)
                if index.is_up:
                    dummy_data.append((index, pos1, other_pos1, pos2, other_pos2))
                else:
                    dummy_data.append((-index, other_pos1, pos1, other_pos2, pos2))
                indices.append(index)
            elif index in free2pos1:
                raise ValueError('Repeated index: %s' % index)
            else:
                free2pos1[index] = pos1
                free2pos2[index] = pos2
                indices.append(index)
            pos2 += 1
    free = [(i, p) for i, p in free2pos2.items()]
    free_names = [i.name for i in free2pos2.keys()]
    dummy_data.sort(key=lambda x: x[3])
    return (indices, free, free_names, dummy_data)