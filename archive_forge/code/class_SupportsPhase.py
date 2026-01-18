from typing import Any, TypeVar
from typing_extensions import Protocol
from cirq._doc import doc_private
class SupportsPhase(Protocol):
    """An effect that can be phased around the Z axis of target qubits."""

    @doc_private
    def _phase_by_(self, phase_turns: float, qubit_index: int):
        """Returns a phased version of the effect.

        Specifically, returns an object with matrix P U P^-1 (up to global
        phase) where U is the given object's matrix and
        P = Z(qubit_index)**(2 * phase_turns). For example, an X gate phased
        by 90 degrees would be a Y gate.

        Args:
            phase_turns: The amount to phase the gate, in fractions of a whole
                turn. Multiply by 2π to get radians.
            qubit_index: The index of the target qubit the phasing applies to.
        Returns:
            The phased gate or operation.
        """