from __future__ import annotations
from itertools import product
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
def _get_rotation_coefficients(self) -> dict[tuple[int, ...], float]:
    """Compute the coefficient of each monomial.

        Returns:
            A dictionary with pairs ``{control_state: rotation angle}`` where ``control_state``
            is a tuple of ``0`` or ``1`` bits.
        """
    all_combinations = list(product([0, 1], repeat=self.num_state_qubits))
    valid_combinations = []
    for combination in all_combinations:
        if 0 < sum(combination) <= self.degree:
            valid_combinations += [combination]
    rotation_coeffs = {control_state: 0.0 for control_state in valid_combinations}
    for i, coeff in enumerate(self.coeffs[1:]):
        i += 1
        for comb, num_combs in _multinomial_coefficients(self.num_state_qubits, i).items():
            control_state: tuple[int, ...] = ()
            power = 1
            for j, qubit in enumerate(comb):
                if qubit > 0:
                    control_state += (1,)
                    power *= 2 ** (j * qubit)
                else:
                    control_state += (0,)
            rotation_coeffs[control_state] += coeff * num_combs * power
    return rotation_coeffs