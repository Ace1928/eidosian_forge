from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.exceptions import CircuitError
from ..boolean_logic import OR
from ..blueprintcircuit import BlueprintCircuit
def _get_twos_complement(self) -> list[int]:
    """Returns the 2's complement of ``self.value`` as array.

        Returns:
             The 2's complement of ``self.value``.
        """
    twos_complement = pow(2, self.num_state_qubits) - int(np.ceil(self.value))
    twos_complement = f'{twos_complement:b}'.rjust(self.num_state_qubits, '0')
    twos_complement = [1 if twos_complement[i] == '1' else 0 for i in reversed(range(len(twos_complement)))]
    return twos_complement