from __future__ import annotations
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import UnitaryGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.basis import BasisTranslator, UnrollCustomDefinitions
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from . import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from ._utils import _ctrl_state_to_int
def add_control(operation: Gate | ControlledGate, num_ctrl_qubits: int, label: str | None, ctrl_state: str | int | None) -> ControlledGate:
    """For standard gates, if the controlled version already exists in the
    library, it will be returned (e.g. XGate.control() = CnotGate().

    For more generic gates, this method implements the controlled
    version by first decomposing into the ['u1', 'u3', 'cx'] basis, then
    controlling each gate in the decomposition.

    Open controls are implemented by conjugating the control line with
    X gates. Adds num_ctrl_qubits controls to operation.

    This function is meant to be called from the
    :method:`qiskit.circuit.gate.Gate.control()` method.

    Args:
        operation: The operation to be controlled.
        num_ctrl_qubits: The number of controls to add to gate.
        label: An optional gate label.
        ctrl_state: The control state in decimal or as a bitstring
            (e.g. '111'). If specified as a bitstring the length
            must equal num_ctrl_qubits, MSB on left. If None, use
            2**num_ctrl_qubits-1.

    Returns:
        Controlled version of gate.

    """
    if isinstance(operation, UnitaryGate):
        operation._define()
    cgate = control(operation, num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
    if operation.label is not None:
        cgate.base_gate = cgate.base_gate.to_mutable()
        cgate.base_gate.label = operation.label
    return cgate