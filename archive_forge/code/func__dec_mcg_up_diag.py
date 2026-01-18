import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from .uc import UCGate
def _dec_mcg_up_diag(self):
    """
        Call to create a circuit with gates that implement the MCG up to a diagonal gate.
        Remark: The qubits the gate acts on are ordered in the following way:
            q=[q_target,q_controls,q_ancilla_zero,q_ancilla_dirty]
        """
    diag = np.ones(2 ** (self.num_controls + 1)).tolist()
    q = QuantumRegister(self.num_qubits)
    circuit = QuantumCircuit(q)
    q_target, q_controls, q_ancillas_zero, q_ancillas_dirty = self._define_qubit_role(q)
    threshold = float('inf')
    if self.num_controls < threshold:
        gate_list = [np.eye(2, 2) for i in range(2 ** self.num_controls)]
        gate_list[-1] = self.params[0]
        ucg = UCGate(gate_list, up_to_diagonal=True)
        circuit.append(ucg, [q_target] + q_controls)
        diag = ucg._get_diagonal()
    return (circuit, diag)