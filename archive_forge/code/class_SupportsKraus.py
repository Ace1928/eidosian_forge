from typing import Any, Sequence, Tuple, TypeVar, Union
import warnings
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.mixture_protocol import has_mixture
from cirq.type_workarounds import NotImplementedType
class SupportsKraus(Protocol):
    """An object that may be describable as a quantum channel."""

    @doc_private
    def _kraus_(self) -> Union[Sequence[np.ndarray], NotImplementedType]:
        """A list of Kraus matrices describing the quantum channel.

        These matrices are the terms in the operator sum representation of a
        quantum channel. If the returned matrices are ${A_0,A_1,..., A_{r-1}}$,
        then this describes the channel:
            $$
            \\rho \\rightarrow \\sum_{k=0}^{r-1} A_k \\rho A_k^\\dagger
            $$
        These matrices are required to satisfy the trace preserving condition
            $$
            \\sum_{k=0}^{r-1} A_k^\\dagger A_k = I
            $$
        where $I$ is the identity matrix. The matrices $A_k$ are sometimes
        called Kraus or noise operators.

        This method is used by the global `cirq.channel` method. If this method
        or the _unitary_ method is not present, or returns NotImplement,
        it is assumed that the receiving object doesn't have a channel
        (resulting in a TypeError or default result when calling `cirq.channel`
        on it). (The ability to return NotImplemented is useful when a class
        cannot know if it is a channel until runtime.)

        The order of cells in the matrices is always implicit with respect to
        the object being called. For example, for GateOperations these matrices
        must be ordered with respect to the list of qubits that the channel is
        applied to. The qubit-to-amplitude order mapping matches the
        ordering of numpy.kron(A, B), where A is a qubit earlier in the list
        than the qubit B.

        Returns:
            A list of matrices describing the channel (Kraus operators), or
            NotImplemented if there is no such matrix.
        """

    @doc_private
    def _has_kraus_(self) -> bool:
        """Whether this value has a Kraus representation.

        This method is used by the global `cirq.has_channel` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to similarly checking `cirq.has_mixture` or `cirq.has_unitary`. If none
        of these are present or return NotImplemented, then `cirq.has_channel`
        will fall back to checking whether `cirq.channel` has a non-default
        value. Otherwise `cirq.has_channel` returns False.

        Returns:
            True if the value has a channel representation, False otherwise.
        """