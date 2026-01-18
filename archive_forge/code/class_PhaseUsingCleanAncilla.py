import dataclasses
import cirq
import numpy as np
from cirq import ops, qis, protocols
@dataclasses.dataclass(frozen=True)
class PhaseUsingCleanAncilla(ops.Gate):
    """Phases the state $|phase_state>$ by $\\exp(1j * \\pi * \\theta)$ using one clean ancilla."""
    theta: float
    phase_state: int = 1
    target_bitsize: int = 1
    ancilla_bitsize: int = 1

    def _num_qubits_(self):
        return self.target_bitsize

    def _decompose_with_context_(self, qubits, *, context: protocols.DecompositionContext):
        anc = context.qubit_manager.qalloc(self.ancilla_bitsize)
        cv = [int(x) for x in f'{self.phase_state:0{self.target_bitsize}b}']
        cnot_ladder = [cirq.CNOT(anc[i - 1], anc[i]) for i in range(1, self.ancilla_bitsize)]
        yield ops.X(anc[0]).controlled_by(*qubits, control_values=cv)
        yield [cnot_ladder, ops.Z(anc[-1]) ** self.theta, reversed(cnot_ladder)]
        yield ops.X(anc[0]).controlled_by(*qubits, control_values=cv)

    def narrow_unitary(self) -> np.ndarray:
        """Narrowed unitary corresponding to the unitary effect applied on target qubits."""
        phase = np.exp(1j * np.pi * self.theta)
        return _matrix_for_phasing_state(self.target_bitsize, self.phase_state, phase)