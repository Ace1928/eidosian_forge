import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
@value.value_equality
class AsymmetricDepolarizingChannel(raw_types.Gate):
    """A channel that depolarizes asymmetrically along different directions.

    This channel applies one of $4^n$ disjoint possibilities: nothing (the
    identity channel) or one of the $4^n - 1$ pauli gates.

    This channel evolves a density matrix via

    $$
    \\sum_i p_i P_i \\rho P_i
    $$

    where i varies from $0$ to $4^n-1$ and $P_i$ represents n-qubit Pauli operator
    (including identity). The input $\\rho$ is the density matrix before the
    depolarization.

    Note: prior to Cirq v0.14, this class contained `num_qubits` as a property.
    This violates the contract of `cirq.Gate` so it was removed.  One can
    instead get the number of qubits by calling the method `num_qubits`.
    """

    def __init__(self, p_x: Optional[float]=None, p_y: Optional[float]=None, p_z: Optional[float]=None, error_probabilities: Optional[Dict[str, float]]=None, tol: float=1e-08) -> None:
        """The asymmetric depolarizing channel.

        Args:
            p_x: The probability that a Pauli X and no other gate occurs.
            p_y: The probability that a Pauli Y and no other gate occurs.
            p_z: The probability that a Pauli Z and no other gate occurs.
            error_probabilities: Dictionary of string (Pauli operator) to its
                probability. If the identity is missing from the list, it will
                be added so that the total probability mass is 1.
            tol: The tolerance used making sure the total probability mass is
                equal to 1.

        Examples of calls:
            * Single qubit: AsymmetricDepolarizingChannel(0.2, 0.1, 0.3)
            * Single qubit: AsymmetricDepolarizingChannel(p_z=0.3)
            * Two qubits: AsymmetricDepolarizingChannel(
                                error_probabilities={'XX': 0.2})

        Raises:
            ValueError: if the args or the sum of args are not probabilities.
        """
        if error_probabilities:
            num_qubits = len(list(error_probabilities)[0])
            for k in error_probabilities.keys():
                if not set(k).issubset({'I', 'X', 'Y', 'Z'}):
                    raise ValueError(f'{k} is not made solely of I, X, Y, Z.')
                if len(k) != num_qubits:
                    raise ValueError(f'{k} must have {num_qubits} Pauli gates.')
            for k, v in error_probabilities.items():
                value.validate_probability(v, f'p({k})')
            sum_probs = sum(error_probabilities.values())
            identity = 'I' * num_qubits
            if sum_probs < 1.0 - tol and identity not in error_probabilities:
                error_probabilities[identity] = 1.0 - sum_probs
            elif abs(sum_probs - 1.0) > tol:
                raise ValueError(f'Probabilities do not add up to 1 but to {sum_probs}')
            self._num_qubits = num_qubits
            self._error_probabilities = error_probabilities
        else:
            p_x = 0.0 if p_x is None else p_x
            p_y = 0.0 if p_y is None else p_y
            p_z = 0.0 if p_z is None else p_z
            p_x = value.validate_probability(p_x, 'p_x')
            p_y = value.validate_probability(p_y, 'p_y')
            p_z = value.validate_probability(p_z, 'p_z')
            p_i = 1 - value.validate_probability(p_x + p_y + p_z, 'p_x + p_y + p_z')
            self._num_qubits = 1
            self._error_probabilities = {'I': p_i, 'X': p_x, 'Y': p_y, 'Z': p_z}

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        ps = []
        for pauli in self._error_probabilities:
            Pi = np.identity(1)
            for gate in pauli:
                if gate == 'I':
                    Pi = np.kron(Pi, protocols.unitary(identity.I))
                elif gate == 'X':
                    Pi = np.kron(Pi, protocols.unitary(pauli_gates.X))
                elif gate == 'Y':
                    Pi = np.kron(Pi, protocols.unitary(pauli_gates.Y))
                elif gate == 'Z':
                    Pi = np.kron(Pi, protocols.unitary(pauli_gates.Z))
            ps.append(Pi)
        return tuple(zip(self._error_probabilities.values(), ps))

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return (self._num_qubits, hash(tuple(sorted(self._error_probabilities.items()))))

    def __repr__(self) -> str:
        return 'cirq.asymmetric_depolarize(' + f'error_probabilities={self._error_probabilities})'

    def __str__(self) -> str:
        return 'asymmetric_depolarize(' + f'error_probabilities={self._error_probabilities})'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> Union[str, Iterable[str]]:
        if self._num_qubits == 1:
            if args.precision is not None:
                return f'A({self.p_x:.{args.precision}g},' + f'{self.p_y:.{args.precision}g},' + f'{self.p_z:.{args.precision}g})'
            return f'A({self.p_x},{self.p_y},{self.p_z})'
        if args.precision is not None:
            error_probabilities = [f'{pauli}:{p:.{args.precision}g}' for pauli, p in self._error_probabilities.items()]
        else:
            error_probabilities = [f'{pauli}:{p}' for pauli, p in self._error_probabilities.items()]
        return [f'A({', '.join(error_probabilities)})'] + [f'({i})' for i in range(1, self._num_qubits)]

    @property
    def p_i(self) -> float:
        """The probability that an Identity I and no other gate occurs."""
        if self._num_qubits != 1:
            raise ValueError('num_qubits should be 1')
        return self._error_probabilities.get('I', 0.0)

    @property
    def p_x(self) -> float:
        """The probability that a Pauli X and no other gate occurs."""
        if self._num_qubits != 1:
            raise ValueError('num_qubits should be 1')
        return self._error_probabilities.get('X', 0.0)

    @property
    def p_y(self) -> float:
        """The probability that a Pauli Y and no other gate occurs."""
        if self._num_qubits != 1:
            raise ValueError('num_qubits should be 1')
        return self._error_probabilities.get('Y', 0.0)

    @property
    def p_z(self) -> float:
        """The probability that a Pauli Z and no other gate occurs."""
        if self._num_qubits != 1:
            raise ValueError('num_qubits should be 1')
        return self._error_probabilities.get('Z', 0.0)

    @property
    def error_probabilities(self) -> Dict[str, float]:
        """A dictionary from Pauli gates to probability"""
        return self._error_probabilities

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['error_probabilities'])

    def _approx_eq_(self, other: Any, atol: float) -> bool:
        self_keys, self_values = zip(*sorted(self.error_probabilities.items()))
        other_keys, other_values = zip(*sorted(other.error_probabilities.items()))
        return self_keys == other_keys and protocols.approx_eq(self_values, other_values, atol=atol)