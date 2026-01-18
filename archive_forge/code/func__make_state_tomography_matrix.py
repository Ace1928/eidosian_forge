from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, protocols, study
from cirq.experiments.qubit_characterizations import TomographyResult
def _make_state_tomography_matrix(self, qubits: Sequence['cirq.Qid']) -> np.ndarray:
    """Gets the matrix used for solving the linear system of the tomography.

        Args:
            qubits: Qubits to do the tomography on.

        Returns:
            A matrix of dimension ((number of rotations)**n * 2**n, 4**n)
            where each column corresponds to the coefficient of a term in the
            density matrix.  Each row is one equation corresponding to a
            rotation sequence and bit string outcome for that rotation sequence.
        """
    num_rots = len(self.rot_sweep)
    num_states = 2 ** self.num_qubits
    unitaries = np.array([protocols.resolve_parameters(self.rot_circuit, rots).unitary(qubit_order=qubits) for rots in self.rot_sweep])
    mat = np.einsum('jkm,jkn->jkmn', unitaries, unitaries.conj())
    return mat.reshape((num_rots * num_states, num_states * num_states))