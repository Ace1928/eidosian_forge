import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
@value.value_equality
class GeneralizedAmplitudeDampingChannel(raw_types.Gate):
    """Dampen qubit amplitudes through non ideal dissipation.

    This channel models the effect of energy dissipation into the environment
    as well as the environment depositing energy into the system.

    Construct a channel to model energy dissipation into the environment
    as well as the environment depositing energy into the system. The
    probabilities with which the energy exchange occur are given by `gamma`,
    and the probability of the environment being not excited is given by
    `p`.

    The stationary state of this channel is the diagonal density matrix
    with probability `p` of being |0⟩ and probability `1-p` of being |1⟩.

    This channel evolves a density matrix via

    $$
    \\rho \\rightarrow \\sum_{i=0}^3 M_i \\rho M_i^\\dagger
    $$

    with

    $$
    \\begin{aligned}
    M_0 =& \\sqrt{p} \\begin{bmatrix}
                        1 & 0  \\\\
                        0 & \\sqrt{1 - \\gamma}
                    \\end{bmatrix}
    \\\\
    M_1 =& \\sqrt{p} \\begin{bmatrix}
                        0 & \\sqrt{\\gamma} \\\\
                        0 & 0
                   \\end{bmatrix}
    \\\\
    M_2 =& \\sqrt{1-p} \\begin{bmatrix}
                        \\sqrt{1-\\gamma} & 0 \\\\
                         0 & 1
                      \\end{bmatrix}
    \\\\
    M_3 =& \\sqrt{1-p} \\begin{bmatrix}
                         0 & 0 \\\\
                         \\sqrt{\\gamma} & 0
                     \\end{bmatrix}
    \\end{aligned}
    $$
    """

    def __init__(self, p: float, gamma: float) -> None:
        """The generalized amplitude damping channel.

        Args:
            p: the probability of the environment being not excited
            gamma: the probability of energy transfer

        Raises:
            ValueError: if gamma or p is not a valid probability.
        """
        self._p = value.validate_probability(p, 'p')
        self._gamma = value.validate_probability(gamma, 'gamma')

    def _num_qubits_(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[np.ndarray]:
        p0 = np.sqrt(self._p)
        p1 = np.sqrt(1.0 - self._p)
        sqrt_g = np.sqrt(self._gamma)
        sqrt_g1 = np.sqrt(1.0 - self._gamma)
        return (p0 * np.array([[1.0, 0.0], [0.0, sqrt_g1]]), p0 * np.array([[0.0, sqrt_g], [0.0, 0.0]]), p1 * np.array([[sqrt_g1, 0.0], [0.0, 1.0]]), p1 * np.array([[0.0, 0.0], [sqrt_g, 0.0]]))

    def _has_kraus_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return (self._p, self._gamma)

    def __repr__(self) -> str:
        return f'cirq.generalized_amplitude_damp(p={self._p!r},gamma={self._gamma!r})'

    def __str__(self) -> str:
        return f'generalized_amplitude_damp(p={self._p!r},gamma={self._gamma!r})'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return f'GAD({f},{f})'.format(self._p, self._gamma)
        return f'GAD({self._p!r},{self._gamma!r})'

    @property
    def p(self) -> float:
        """The probability of the environment being not excited."""
        return self._p

    @property
    def gamma(self) -> float:
        """The probability of energy transfer."""
        return self._gamma

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p', 'gamma'])

    def _approx_eq_(self, other: Any, atol: float) -> bool:
        return np.isclose(self.gamma, other.gamma, atol=atol).item() and np.isclose(self.p, other.p, atol=atol).item()