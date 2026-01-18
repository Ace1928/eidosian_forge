from __future__ import annotations
from collections.abc import Callable
from qiskit import circuit
from qiskit.circuit import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from ..standard_gates import XGate, YGate, ZGate, HGate, TGate, TdgGate, SGate, SdgGate
def _ccx_v_chain_rule(self, control_qubits: QuantumRegister | list[circuit.Qubit], ancilla_qubits: QuantumRegister | list[circuit.Qubit], reverse: bool=False) -> None:
    """Get the rule for the CCX V-chain.

        The CCX V-chain progressively computes the CCX of the control qubits and puts the final
        result in the last ancillary qubit.

        Args:
            control_qubits: The control qubits.
            ancilla_qubits: The ancilla qubits.
            reverse: If True, compute the chain down to the qubit. If False, compute upwards.

        Returns:
            The rule for the (reversed) CCX V-chain.

        Raises:
            QiskitError: If an insufficient number of ancilla qubits was provided.
        """
    if len(ancilla_qubits) == 0:
        return
    if len(ancilla_qubits) < len(control_qubits) - 1:
        raise QiskitError('Insufficient number of ancilla qubits.')
    iterations = list(enumerate(range(2, len(control_qubits))))
    if not reverse:
        self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])
        for i, j in iterations:
            self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
    else:
        for i, j in reversed(iterations):
            self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
        self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])