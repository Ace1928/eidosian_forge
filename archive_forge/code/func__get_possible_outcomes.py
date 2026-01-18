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
def _get_possible_outcomes(m, bits):
    """Get the possible states that can be produced in a measurement.

    Parameters
    ----------
    m : Matrix
        The matrix representing the state of the system.
    bits : tuple, list
        Which bits will be measured.

    Returns
    -------
    result : list
        The list of possible states which can occur given this measurement.
        These are un-normalized so we can derive the probability of finding
        this state by taking the inner product with itself
    """
    size = max(m.shape)
    nqubits = int(math.log(size, 2) + 0.1)
    output_matrices = []
    for i in range(1 << len(bits)):
        output_matrices.append(zeros(2 ** nqubits, 1))
    bit_masks = []
    for bit in bits:
        bit_masks.append(1 << bit)
    for i in range(2 ** nqubits):
        trueness = 0
        for j in range(len(bit_masks)):
            if i & bit_masks[j]:
                trueness += j + 1
        output_matrices[trueness][i] = m[i]
    return output_matrices