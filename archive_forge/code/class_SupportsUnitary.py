from typing import Any, TypeVar, Union, Optional
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import ApplyUnitaryArgs, apply_unitaries
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.type_workarounds import NotImplementedType
class SupportsUnitary(Protocol):
    """An object that may be describable by a unitary matrix."""

    @doc_private
    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        """A unitary matrix describing this value, e.g. the matrix of a gate.

        This method is used by the global `cirq.unitary` method. If this method
        is not present, or returns NotImplemented, it is assumed that the
        receiving object doesn't have a unitary matrix (resulting in a TypeError
        or default result when calling `cirq.unitary` on it). (The ability to
        return NotImplemented is useful when a class cannot know if it has a
        matrix until runtime, e.g. cirq.X**c normally has a matrix but
        cirq.X**sympy.Symbol('a') doesn't.)

        The order of cells in the matrix is always implicit with respect to the
        object being called. For example, for gates the matrix must be ordered
        with respect to the list of qubits that the gate is applied to. For
        operations, the matrix is ordered to match the list returned by its
        `qubits` attribute. The qubit-to-amplitude order mapping matches the
        ordering of numpy.kron(A, B), where A is a qubit earlier in the list
        than the qubit B.

        Returns:
            A unitary matrix describing this value, or NotImplemented if there
            is no such matrix.
        """

    @doc_private
    def _has_unitary_(self) -> bool:
        """Whether this value has a unitary matrix representation.

        This method is used by the global `cirq.has_unitary` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using _unitary_ with a default value, or False if neither exist.

        Returns:
            True if the value has a unitary matrix representation, False
            otherwise.
        """