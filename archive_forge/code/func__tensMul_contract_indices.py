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
def _tensMul_contract_indices(args, replace_indices=True):
    replacements = [{} for _ in args]
    args_indices = [get_indices(arg) for arg in args]
    indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)
    cdt = defaultdict(int)

    def dummy_name_gen(tensor_index_type):
        nd = str(cdt[tensor_index_type])
        cdt[tensor_index_type] += 1
        return tensor_index_type.dummy_name + '_' + nd
    if replace_indices:
        for old_index, pos1cov, pos1contra, pos2cov, pos2contra in dummy_data:
            index_type = old_index.tensor_index_type
            while True:
                dummy_name = dummy_name_gen(index_type)
                if dummy_name not in free_names:
                    break
            dummy = TensorIndex(dummy_name, index_type, True)
            replacements[pos1cov][old_index] = dummy
            replacements[pos1contra][-old_index] = -dummy
            indices[pos2cov] = dummy
            indices[pos2contra] = -dummy
        args = [arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg, repl in zip(args, replacements)]
    dum = TensMul._dummy_data_to_dum(dummy_data)
    return (args, indices, free, dum)