from typing import AbstractSet, Any, Dict, Union
import numpy as np
import sympy
import cirq
from cirq import value, _compat
from cirq.ops import raw_types
@value.value_equality
class QuantumFourierTransformGate(raw_types.Gate):
    """Switches from the computational basis to the frequency basis.

    This gate has the unitary

    $$
    \\frac{1}{2^{n/2}}\\sum_{x,y=0}^{2^n-1} \\omega^{xy} |x\\rangle\\langle y|
    $$

    where
    $$
    \\omega = e^{\\frac{2\\pi i}{2^n}}
    $$
    """

    def __init__(self, num_qubits: int, *, without_reverse: bool=False):
        """Inits QuantumFourierTransformGate.

        Args:
            num_qubits: The number of qubits the gate applies to.
            without_reverse: Whether or not to include the swaps at the end
                of the circuit decomposition that reverse the order of the
                qubits. These are technically necessary in order to perform the
                correct effect, but can almost always be optimized away by just
                performing later operations on different qubits.
        """
        self._num_qubits = num_qubits
        self._without_reverse = without_reverse

    def _json_dict_(self) -> Dict[str, Any]:
        return {'num_qubits': self._num_qubits, 'without_reverse': self._without_reverse}

    def _value_equality_values_(self):
        return (self._num_qubits, self._without_reverse)

    def num_qubits(self) -> int:
        return self._num_qubits

    def _decompose_(self, qubits):
        if len(qubits) == 0:
            return
        yield cirq.H(qubits[0])
        for i in range(1, len(qubits)):
            yield PhaseGradientGate(num_qubits=i, exponent=0.5).on(*qubits[:i][::-1]).controlled_by(qubits[i])
            yield cirq.H(qubits[i])
        if not self._without_reverse:
            for i in range(len(qubits) // 2):
                yield cirq.SWAP(qubits[i], qubits[-i - 1])

    def _has_unitary_(self):
        return True

    def __str__(self) -> str:
        return 'qft[norev]' if self._without_reverse else 'qft'

    def __repr__(self) -> str:
        return f'cirq.QuantumFourierTransformGate(num_qubits={self._num_qubits!r}, without_reverse={self._without_reverse!r})'

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        return cirq.CircuitDiagramInfo(wire_symbols=(str(self),) + tuple((f'#{k + 1}' for k in range(1, self._num_qubits))), exponent_qubit_index=0)