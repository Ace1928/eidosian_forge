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
def _correct_signature_from_indices(data, indices, free, dum, inverse=False):
    """
        Utility function to correct the values inside the components data
        ndarray according to whether indices are covariant or contravariant.

        It uses the metric matrix to lower values of covariant indices.
        """
    for i, indx in enumerate(indices):
        if not indx.is_up and (not inverse):
            data = _TensorDataLazyEvaluator._flip_index_by_metric(data, indx.tensor_index_type.data, i)
        elif not indx.is_up and inverse:
            data = _TensorDataLazyEvaluator._flip_index_by_metric(data, _TensorDataLazyEvaluator.inverse_matrix(indx.tensor_index_type.data), i)
    return data