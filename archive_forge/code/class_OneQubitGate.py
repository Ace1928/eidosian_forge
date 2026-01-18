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
class OneQubitGate(Gate):
    """A single qubit unitary gate base class."""
    nqubits = _S.One

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(self.gate_name_plot, gate_idx, int(self.targets[0]))

    def _eval_commutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return _S.Zero
        return Operator._eval_commutator(self, other, **hints)

    def _eval_anticommutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return Integer(2) * self * other
        return Operator._eval_anticommutator(self, other, **hints)