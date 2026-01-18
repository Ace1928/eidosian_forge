from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def _get_valued_base_test_variables():
    minkowski = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)))
    Lorentz = TensorIndexType('Lorentz', dim=4)
    Lorentz.data = minkowski
    i0, i1, i2, i3, i4 = tensor_indices('i0:5', Lorentz)
    E, px, py, pz = symbols('E px py pz')
    A = TensorHead('A', [Lorentz])
    A.data = [E, px, py, pz]
    B = TensorHead('B', [Lorentz], TensorSymmetry.no_symmetry(1), 'Gcomm')
    B.data = range(4)
    AB = TensorHead('AB', [Lorentz] * 2)
    AB.data = minkowski
    ba_matrix = Matrix(((1, 2, 3, 4), (5, 6, 7, 8), (9, 0, -1, -2), (-3, -4, -5, -6)))
    BA = TensorHead('BA', [Lorentz] * 2)
    BA.data = ba_matrix
    LorentzD = TensorIndexType('LorentzD')
    LorentzD.data = [-1, 1, 1, 1]
    mu0, mu1, mu2 = tensor_indices('mu0:3', LorentzD)
    C = TensorHead('C', [LorentzD])
    C.data = [E, px, py, pz]
    ndm_matrix = ((1, 1, 0), (1, 0, 1), (0, 1, 0))
    ndm = TensorIndexType('ndm')
    ndm.data = ndm_matrix
    n0, n1, n2 = tensor_indices('n0:3', ndm)
    NA = TensorHead('NA', [ndm])
    NA.data = range(10, 13)
    NB = TensorHead('NB', [ndm] * 2)
    NB.data = [[i + j for j in range(10, 13)] for i in range(10, 13)]
    NC = TensorHead('NC', [ndm] * 3)
    NC.data = [[[i + j + k for k in range(4, 7)] for j in range(1, 4)] for i in range(2, 5)]
    return (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1, n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4)