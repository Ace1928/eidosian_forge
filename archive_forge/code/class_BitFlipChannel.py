import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
@value.value_equality
class BitFlipChannel(raw_types.Gate):
    """Probabilistically flip a qubit from 1 to 0 state or vice versa.

    Construct a channel that flips a qubit with probability p.

    This channel evolves a density matrix via:

    $$
    \\rho \\rightarrow M_0 \\rho M_0^\\dagger + M_1 \\rho M_1^\\dagger
    $$

    With:

    $$
    \\begin{aligned}
        M_0 =& \\sqrt{1 - p} \\begin{bmatrix}
                            1 & 0  \\\\
                            0 & 1
                       \\end{bmatrix}
        \\\\
        M_1 =& \\sqrt{p} \\begin{bmatrix}
                            0 & 1 \\\\
                            1 & 0
                         \\end{bmatrix}
        \\end{aligned}
    $$
    """

    def __init__(self, p: float) -> None:
        """Construct a channel that probabilistically flips a qubit.

        Args:
            p: the probability of a bit flip.

        Raises:
            ValueError: if p is not a valid probability.
        """
        self._p = value.validate_probability(p, 'p')
        self._delegate = AsymmetricDepolarizingChannel(p, 0.0, 0.0)

    def _num_qubits_(self) -> int:
        return 1

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        mixture = self._delegate._mixture_()
        return (mixture[0], mixture[1])

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return f'cirq.bit_flip(p={self._p!r})'

    def __str__(self) -> str:
        return f'bit_flip(p={self._p!r})'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return f'BF({f})'.format(self._p)
        return f'BF({self._p!r})'

    @property
    def p(self) -> float:
        """The probability of a bit flip."""
        return self._p

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p'])

    def _approx_eq_(self, other: Any, atol: float) -> bool:
        return np.isclose(self._p, other._p, atol=atol).item()