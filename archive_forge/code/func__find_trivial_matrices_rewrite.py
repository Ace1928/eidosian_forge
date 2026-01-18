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
def _find_trivial_matrices_rewrite(expr: ArrayTensorProduct):
    trivial_matrices = []
    pos: Optional[int] = None
    first: Optional[MatrixExpr] = None
    second: Optional[MatrixExpr] = None
    removed: List[int] = []
    counter: int = 0
    args: List[Optional[Basic]] = list(expr.args)
    for i, arg in enumerate(expr.args):
        if isinstance(arg, MatrixExpr):
            if arg.shape == (1, 1):
                trivial_matrices.append(arg)
                args[i] = None
                removed.extend([counter, counter + 1])
            elif pos is None and isinstance(arg, MatMul):
                margs = arg.args
                for j, e in enumerate(margs):
                    if isinstance(e, MatrixExpr) and e.shape[1] == 1:
                        pos = i
                        first = MatMul.fromiter(margs[:j + 1])
                        second = MatMul.fromiter(margs[j + 1:])
                        break
        counter += get_rank(arg)
    if pos is None:
        return (expr, [])
    args[pos] = (first * MatMul.fromiter((i for i in trivial_matrices)) * second).doit()
    return (_array_tensor_product(*[i for i in args if i is not None]), removed)