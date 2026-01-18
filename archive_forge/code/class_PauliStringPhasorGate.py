from typing import (
import numbers
import sympy
from cirq import value, protocols
from cirq._compat import proper_repr
from cirq.ops import (
@value.value_equality(approximate=True)
class PauliStringPhasorGate(raw_types.Gate):
    """A gate that phases the eigenstates of a Pauli string.

    The -1 eigenstates of the Pauli string will have their amplitude multiplied
    by e^(i pi exponent_neg) while +1 eigenstates of the Pauli string will have
    their amplitude multiplied by e^(i pi exponent_pos).
    """

    def __init__(self, dense_pauli_string: dps.DensePauliString, *, exponent_neg: Union[int, float, sympy.Expr]=1, exponent_pos: Union[int, float, sympy.Expr]=0) -> None:
        """Initializes the PauliStringPhasorGate.

        Args:
            dense_pauli_string: The DensePauliString defining the positive and
                negative eigenspaces that will be independently phased.
            exponent_neg: How much to phase vectors in the negative eigenspace,
                in the form of the t in (-1)**t = exp(i pi t).
            exponent_pos: How much to phase vectors in the positive eigenspace,
                in the form of the t in (-1)**t = exp(i pi t).

        Raises:
            ValueError: If coefficient is not 1 or -1.
        """
        if dense_pauli_string.coefficient == -1:
            dense_pauli_string = -dense_pauli_string
            exponent_pos, exponent_neg = (exponent_neg, exponent_pos)
        if dense_pauli_string.coefficient != 1:
            raise ValueError("Given DensePauliString doesn't have +1 and -1 eigenvalues. dense_pauli_string.coefficient must be 1 or -1.")
        self._dense_pauli_string = dense_pauli_string
        self._exponent_neg = value.canonicalize_half_turns(exponent_neg)
        self._exponent_pos = value.canonicalize_half_turns(exponent_pos)

    @property
    def exponent_relative(self) -> Union[int, float, sympy.Expr]:
        """The relative exponent between negative and positive exponents."""
        return value.canonicalize_half_turns(self.exponent_neg - self.exponent_pos)

    @property
    def exponent_neg(self) -> Union[int, float, sympy.Expr]:
        """The negative exponent."""
        return self._exponent_neg

    @property
    def exponent_pos(self) -> Union[int, float, sympy.Expr]:
        """The positive exponent."""
        return self._exponent_pos

    @property
    def dense_pauli_string(self) -> 'cirq.DensePauliString':
        """The underlying DensePauliString."""
        return self._dense_pauli_string

    def _value_equality_values_(self):
        return (self.dense_pauli_string, self.exponent_neg, self.exponent_pos)

    def equal_up_to_global_phase(self, other: 'cirq.PauliStringPhasorGate') -> bool:
        """Checks equality of two PauliStringPhasors, up to global phase."""
        if isinstance(other, PauliStringPhasorGate):
            rel1 = self.exponent_relative
            rel2 = other.exponent_relative
            return rel1 == rel2 and self.dense_pauli_string == other.dense_pauli_string
        return False

    def __pow__(self, exponent: Union[float, sympy.Symbol]) -> 'PauliStringPhasorGate':
        pn = protocols.mul(self.exponent_neg, exponent, None)
        pp = protocols.mul(self.exponent_pos, exponent, None)
        if pn is None or pp is None:
            return NotImplemented
        return PauliStringPhasorGate(self.dense_pauli_string, exponent_neg=pn, exponent_pos=pp)

    def _has_unitary_(self) -> bool:
        return not self._is_parameterized_()

    def _to_z_basis_ops(self, qubits: Sequence['cirq.Qid']) -> Iterator[raw_types.Operation]:
        """Returns operations to convert the qubits to the computational basis."""
        return self.dense_pauli_string.on(*qubits).to_z_basis_ops()

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        if len(self.dense_pauli_string) <= 0:
            return
        any_qubit = qubits[0]
        to_z_ops = op_tree.freeze_op_tree(self._to_z_basis_ops(qubits))
        xor_decomp = tuple(xor_nonlocal_decompose(qubits, any_qubit))
        yield to_z_ops
        yield xor_decomp
        if self.exponent_neg:
            yield (pauli_gates.Z(any_qubit) ** self.exponent_neg)
        if self.exponent_pos:
            yield pauli_gates.X(any_qubit)
            yield (pauli_gates.Z(any_qubit) ** self.exponent_pos)
            yield pauli_gates.X(any_qubit)
        yield protocols.inverse(xor_decomp)
        yield protocols.inverse(to_z_ops)

    def _trace_distance_bound_(self) -> float:
        if len(self.dense_pauli_string) == 0:
            return 0.0
        return protocols.trace_distance_bound(pauli_gates.Z ** self.exponent_relative)

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.exponent_neg) or protocols.is_parameterized(self.exponent_pos)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.exponent_neg) | protocols.parameter_names(self.exponent_pos)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'PauliStringPhasorGate':
        exponent_neg = resolver.value_of(self.exponent_neg, recursive)
        exponent_pos = resolver.value_of(self.exponent_pos, recursive)
        if isinstance(exponent_neg, (complex, numbers.Complex)):
            if isinstance(exponent_neg, numbers.Real):
                exponent_neg = float(exponent_neg)
            else:
                raise ValueError(f'PauliStringPhasorGate does not support complex exponent {exponent_neg}')
        if isinstance(exponent_pos, (complex, numbers.Complex)):
            if isinstance(exponent_pos, numbers.Real):
                exponent_pos = float(exponent_pos)
            else:
                raise ValueError(f'PauliStringPhasorGate does not support complex exponent {exponent_pos}')
        return PauliStringPhasorGate(self.dense_pauli_string, exponent_neg=exponent_neg, exponent_pos=exponent_pos)

    def __repr__(self) -> str:
        return f'cirq.PauliStringPhasorGate({self.dense_pauli_string!r}, exponent_neg={proper_repr(self.exponent_neg)}, exponent_pos={proper_repr(self.exponent_pos)})'

    def __str__(self) -> str:
        if self.exponent_pos == -self.exponent_neg:
            sign = '-' if self.exponent_pos < 0 else ''
            exponent = str(abs(self.exponent_pos))
            return f'exp({sign}iπ{exponent}*{self.dense_pauli_string})'
        return f'({self.dense_pauli_string})**{self.exponent_relative}'

    def num_qubits(self) -> int:
        """The number of qubits for the gate."""
        return len(self.dense_pauli_string)

    def on(self, *qubits: 'cirq.Qid') -> 'cirq.PauliStringPhasor':
        """Creates a PauliStringPhasor on the qubits."""
        return PauliStringPhasor(self.dense_pauli_string.on(*qubits), qubits=qubits, exponent_pos=self.exponent_pos, exponent_neg=self.exponent_neg)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['dense_pauli_string', 'exponent_neg', 'exponent_pos'])