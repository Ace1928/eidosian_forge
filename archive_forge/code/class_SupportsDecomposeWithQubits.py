import itertools
import dataclasses
import inspect
from collections import defaultdict
from typing import (
from typing_extensions import runtime_checkable
from typing_extensions import Protocol
from cirq import devices, ops
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
class SupportsDecomposeWithQubits(Protocol):
    """An object that can be decomposed into operations on given qubits.

    Returning `NotImplemented` or `None` means "not decomposable". Otherwise an
    operation, list of operations, or generally anything meeting the `OP_TREE`
    contract can be returned.

    For example, a SWAP gate can be turned into three CNOTs. But in order to
    describe those CNOTs one must be able to talk about "the target qubit" and
    "the control qubit". This can only be done once the qubits-to-be-swapped are
    known.

    The main user of this protocol is `GateOperation`, which decomposes itself
    by delegating to its gate. The qubits argument is needed because gates are
    specified independently of target qubits and so must be told the relevant
    qubits. A `GateOperation` implements `SupportsDecompose` as long as its gate
    implements `SupportsDecomposeWithQubits`.
    """

    def _decompose_(self, qubits: Tuple['cirq.Qid', ...]) -> DecomposeResult:
        pass

    def _decompose_with_context_(self, qubits: Tuple['cirq.Qid', ...], *, context: Optional[DecompositionContext]=None) -> DecomposeResult:
        pass