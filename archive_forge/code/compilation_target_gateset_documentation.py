from typing import Optional, List, Hashable, TYPE_CHECKING
import abc
from cirq import circuits, ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import merge_k_qubit_gates, merge_single_qubit_gates
Decomposes (connected component of) 2-qubit operations using gates from this gateset.

        Args:
            op: A two-qubit operation (can be a tagged `cirq.CircuitOperation` wrapping
                a connected component of 1 & 2  qubit unitaries).
            moment_idx: Index of the moment in which operation `op` occurs.

        Returns:
            A `cirq.OP_TREE` implementing `op` using gates from this gateset OR
            None or NotImplemented if decomposition of `op` is unknown.
        