from typing import Any, Sequence, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.type_workarounds import NotImplementedType
class SupportsMixture(Protocol):
    """An object that decomposes into a probability distribution of unitaries."""

    @doc_private
    def _mixture_(self) -> Union[Sequence[Tuple[float, Any]], NotImplementedType]:
        """Decompose into a probability distribution of unitaries.

        This method is used by the global `cirq.mixture` method.

        A mixture is described by an iterable of tuples of the form

            (probability of unitary, unitary as numpy array)

        The probability components of the tuples must sum to 1.0 and be between
        0 and 1 (inclusive).

        Returns:
            A list of (probability, unitary) pairs.
        """

    @doc_private
    def _has_mixture_(self) -> bool:
        """Whether this value has a mixture representation.

        This method is used by the global `cirq.has_mixture` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using _mixture_ with a default value, or False if neither exist.

        Returns:
          True if the value has a mixture representation, Falseotherwise.
        """