from typing import Any, TypeVar, Optional, Sequence, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import unitary_protocol
class SupportsTraceDistanceBound(Protocol):
    """An effect with known bounds on how easy it is to detect.

    Used when deciding whether or not an operation is negligible. For example,
    the trace distance between the states before and after a Z**0.00000001
    operation is very close to 0, so it would typically be considered
    negligible.
    """

    @doc_private
    def _trace_distance_bound_(self) -> float:
        """A maximum on the trace distance between `val`'s input and output.

        Generally this method is used when deciding whether to keep gates, so
        only the behavior near 0 is important. Approximations that overestimate
        the maximum trace distance are permitted. If, for any case, the bound
        exceeds 1, this function will return 1.  Underestimates are not
        permitted.
        """