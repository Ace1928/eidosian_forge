import logging
import numpy as np
def _lower_limit(num_qubits: int) -> int:
    """
    Returns lower limit on the number of CNOT units that guarantees exact representation of
    a unitary operator by quantum gates.

    Args:
        num_qubits: number of qubits.

    Returns:
        lower limit on the number of CNOT units.
    """
    num_cnots = round(np.ceil((4 ** num_qubits - 3 * num_qubits - 1) / 4.0))
    return num_cnots