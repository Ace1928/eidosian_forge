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
class UGate(Gate):
    """General gate specified by a set of targets and a target matrix.

    Parameters
    ----------
    label : tuple
        A tuple of the form (targets, U), where targets is a tuple of the
        target qubits and U is a unitary matrix with dimension of
        len(targets).
    """
    gate_name = 'U'
    gate_name_latex = 'U'

    @classmethod
    def _eval_args(cls, args):
        targets = args[0]
        if not is_sequence(targets):
            targets = (targets,)
        targets = Gate._eval_args(targets)
        _validate_targets_controls(targets)
        mat = args[1]
        if not isinstance(mat, MatrixBase):
            raise TypeError('Matrix expected, got: %r' % mat)
        mat = _sympify(mat)
        dim = 2 ** len(targets)
        if not all((dim == shape for shape in mat.shape)):
            raise IndexError('Number of targets must match the matrix size: %r %r' % (targets, mat))
        return (targets, mat)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2) ** (_max(args[0]) + 1)

    @property
    def targets(self):
        """A tuple of target qubits."""
        return tuple(self.label[0])

    def get_target_matrix(self, format='sympy'):
        """The matrix rep. of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        return self.label[1]

    def _pretty(self, printer, *args):
        targets = self._print_sequence_pretty(self.targets, ',', printer, *args)
        gate_name = stringPict(self.gate_name)
        return self._print_subscript_pretty(gate_name, targets)

    def _latex(self, printer, *args):
        targets = self._print_sequence(self.targets, ',', printer, *args)
        return '%s_{%s}' % (self.gate_name_latex, targets)

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(self.gate_name_plot, gate_idx, int(self.targets[0]))