import math
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import log
from sympy.core.basic import _sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import Matrix, zeros
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.state import Ket, Bra, State
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.matrixutils import (
from mpmath.libmp.libintmath import bitcount
@classmethod
def _eval_args_with_nqubits(cls, number, nqubits):
    need = bitcount(abs(number))
    if nqubits < need:
        raise ValueError('cannot represent %s with %s bits' % (number, nqubits))
    qubit_values = [number >> i & 1 for i in reversed(range(nqubits))]
    return QubitState._eval_args(qubit_values)