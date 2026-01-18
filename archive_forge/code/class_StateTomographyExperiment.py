from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, protocols, study
from cirq.experiments.qubit_characterizations import TomographyResult
class StateTomographyExperiment:
    """Experiment to conduct state tomography.

    Generates data collection protocol for the state tomography experiment.
    Does the fitting of generated data to determine the density matrix.

    Attributes:
        rot_circuit: Circuit with parameterized rotation gates to do before the
            final measurements.
        rot_sweep: The list of rotations on the qubits to perform before
            measurement.
        mat: Matrix of coefficients for the system.  Each row is one equation
            corresponding to a rotation sequence and bit string outcome for
            that rotation sequence.  Each column corresponds to the coefficient
            on one term in the density matrix.
        num_qubits: Number of qubits to do tomography on.
    """

    def __init__(self, qubits: Sequence['cirq.Qid'], prerotations: Optional[Sequence[Tuple[float, float]]]=None):
        """Initializes the rotation protocol and matrix for system.

        Args:
            qubits: Qubits to do the tomography on.
            prerotations: Tuples of (phase_exponent, exponent) parameters for
                gates to apply to the qubits before measurement. The actual
                rotation applied will be `cirq.PhasedXPowGate` with the
                specified values of phase_exponent and exponent. If None,
                we use [(0, 0), (0, 0.5), (0.5, 0.5)], which corresponds
                to rotation gates [I, X**0.5, Y**0.5].
        """
        if prerotations is None:
            prerotations = [(0, 0), (0, 0.5), (0.5, 0.5)]
        self.num_qubits = len(qubits)
        phase_exp_vals, exp_vals = zip(*prerotations)
        operations: List['cirq.Operation'] = []
        sweeps: List['cirq.Sweep'] = []
        for i, qubit in enumerate(qubits):
            phase_exp = sympy.Symbol(f'phase_exp_{i}')
            exp = sympy.Symbol(f'exp_{i}')
            gate = ops.PhasedXPowGate(phase_exponent=phase_exp, exponent=exp)
            operations.append(gate.on(qubit))
            sweeps.append(study.Points(phase_exp, phase_exp_vals) + study.Points(exp, exp_vals))
        self.rot_circuit = circuits.Circuit(operations)
        self.rot_sweep = study.Product(*sweeps)
        self.mat = self._make_state_tomography_matrix(qubits)

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

    def fit_density_matrix(self, counts: np.ndarray) -> TomographyResult:
        """Solves equation mat * rho = probs.

        Args:
            counts: A 2D array where each row contains measured counts
                of all n-qubit bitstrings for the corresponding pre-rotations
                in `rot_sweep`.  The order of the probabilities corresponds to
                to `rot_sweep` and the order of the bit strings corresponds to
                increasing integers up to 2**(num_qubits)-1.

        Returns:
            `TomographyResult` with density matrix corresponding to solution of
            this system.
        """
        probs = counts / np.sum(counts, axis=1)[:, np.newaxis]
        c, _, _, _ = np.linalg.lstsq(self.mat, np.asarray(probs).flat, rcond=-1)
        rho = c.reshape((2 ** self.num_qubits, 2 ** self.num_qubits))
        return TomographyResult(rho)