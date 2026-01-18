import numbers
from collections.abc import Iterable
from typing import Any, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.approximate_equality_protocol import approx_eq
class SupportsEqualUpToGlobalPhase(Protocol):
    """Object which can be compared for equality mod global phase."""

    @doc_private
    def _equal_up_to_global_phase_(self, other: Any, *, atol: Union[int, float]) -> bool:
        """Approximate comparator.

        Types implementing this protocol define their own logic for comparison
        with other types.

        Args:
            other: Target object for comparison of equality up to global phase.
            atol: The minimum absolute tolerance. See `np.isclose()`
                documentation for details.

        Returns:
            True if objects are equal up to a global phase, False otherwise.
            Returns NotImplemented when checking equality up to a global phase
            is not implemented for given types.
        """