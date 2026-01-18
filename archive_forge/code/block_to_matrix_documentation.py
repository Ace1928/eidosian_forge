from qiskit.quantum_info import Operator
from qiskit.exceptions import QiskitError
from qiskit._accelerate.convert_2q_block_matrix import blocks_to_matrix

    The function converts any sequence of operations between two qubits into a matrix
    that can be utilized to create a gate or a unitary.

    Args:
        block (List(DAGOpNode)): A block of operations on two qubits.
        block_index_map (dict(Qubit, int)): The mapping of the qubit indices in the main circuit.

    Returns:
        NDArray: Matrix representation of the block of operations.
    