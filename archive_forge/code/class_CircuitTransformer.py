from typing import Dict, cast, Optional, Tuple, List, Callable
from pyquil import Program
import cirq
from cirq_rigetti.quil_output import RigettiQCSQuilOutput
from typing_extensions import Protocol
class CircuitTransformer(Protocol):
    """A type definition for `cirq.Circuit` to `pyquil.Program` transformer functions."""

    def __call__(self, *, circuit: cirq.Circuit) -> Tuple[Program, Dict[str, str]]:
        """Transforms a `cirq.Circuit` to a pyquil.Program`.

        Args:
            circuit: The `cirq.Circuit` the transformer will transform into a `pyquil.Program`.

        Returns:
            The `pyquil.Program` and a map of the `cirq.Circuit`'s memory region keys to
            the `pyquil.Program`'s memory regions.
        """