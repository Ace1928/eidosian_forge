from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import Barrier
def _check_unitary(circuit):
    """Check a circuit is unitary by checking if all operations are of type ``Gate``."""
    for instruction in circuit.data:
        if not isinstance(instruction.operation, (Gate, Barrier)):
            raise CircuitError('One or more instructions cannot be converted to a gate. "{}" is not a gate instruction'.format(instruction.operation.name))