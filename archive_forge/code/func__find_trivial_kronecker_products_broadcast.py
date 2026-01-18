import itertools
from collections import defaultdict
from typing import Tuple as tTuple, Union as tUnion, FrozenSet, Dict as tDict, List, Optional
from functools import singledispatch
from itertools import accumulate
from sympy import MatMul, Basic, Wild, KroneckerProduct
from sympy.assumptions.ask import (Q, ask)
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.matrices.expressions.diagonal import DiagMatrix
from sympy.matrices.expressions.hadamard import hadamard_product, HadamardPower
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import (Identity, ZeroMatrix, OneMatrix)
from sympy.matrices.expressions.trace import Trace
from sympy.matrices.expressions.transpose import Transpose
from sympy.combinatorics.permutations import _af_invert, Permutation
from sympy.matrices.common import MatrixCommon
from sympy.matrices.expressions.applyfunc import ElementwiseApplyFunction
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.tensor.array.expressions.array_expressions import PermuteDims, ArrayDiagonal, \
from sympy.tensor.array.expressions.utils import _get_mapping_from_subranks
def _find_trivial_kronecker_products_broadcast(expr: ArrayTensorProduct):
    newargs: List[Basic] = []
    removed = []
    count_dims = 0
    for i, arg in enumerate(expr.args):
        count_dims += get_rank(arg)
        shape = get_shape(arg)
        current_range = [count_dims - i for i in range(len(shape), 0, -1)]
        if shape == (1, 1) and len(newargs) > 0 and (1 not in get_shape(newargs[-1])) and isinstance(newargs[-1], MatrixExpr) and isinstance(arg, MatrixExpr):
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            removed.extend(current_range)
        elif 1 not in shape and len(newargs) > 0 and (get_shape(newargs[-1]) == (1, 1)):
            newargs[-1] = KroneckerProduct(newargs[-1], arg)
            prev_range = [i for i in range(min(current_range)) if i not in removed]
            removed.extend(prev_range[-2:])
        else:
            newargs.append(arg)
    return (_array_tensor_product(*newargs), removed)