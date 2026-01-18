from itertools import chain
import random
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.power import Pow
from sympy.core.numbers import Number
from sympy.core.singleton import S as _S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import (UnitaryOperator, Operator,
from sympy.physics.quantum.matrixutils import matrix_tensor_product, matrix_eye
from sympy.physics.quantum.matrixcache import matrix_cache
from sympy.matrices.matrices import MatrixBase
from sympy.utilities.iterables import is_sequence
class HadamardGate(HermitianOperator, OneQubitGate):
    """The single qubit Hadamard gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> from sympy.physics.quantum.gate import HadamardGate
    >>> from sympy.physics.quantum.qapply import qapply
    >>> qapply(HadamardGate(0)*Qubit('1'))
    sqrt(2)*|0>/2 - sqrt(2)*|1>/2
    >>> # Hadamard on bell state, applied on 2 qubits.
    >>> psi = 1/sqrt(2)*(Qubit('00')+Qubit('11'))
    >>> qapply(HadamardGate(0)*HadamardGate(1)*psi)
    sqrt(2)*|00>/2 + sqrt(2)*|11>/2

    """
    gate_name = 'H'
    gate_name_latex = 'H'

    def get_target_matrix(self, format='sympy'):
        if _normalized:
            return matrix_cache.get_matrix('H', format)
        else:
            return matrix_cache.get_matrix('Hsqrt2', format)

    def _eval_commutator_XGate(self, other, **hints):
        return I * sqrt(2) * YGate(self.targets[0])

    def _eval_commutator_YGate(self, other, **hints):
        return I * sqrt(2) * (ZGate(self.targets[0]) - XGate(self.targets[0]))

    def _eval_commutator_ZGate(self, other, **hints):
        return -I * sqrt(2) * YGate(self.targets[0])

    def _eval_anticommutator_XGate(self, other, **hints):
        return sqrt(2) * IdentityGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator_ZGate(self, other, **hints):
        return sqrt(2) * IdentityGate(self.targets[0])