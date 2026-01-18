from typing import Optional, TYPE_CHECKING
from cirq import protocols
from cirq.transformers import transformer_api, transformer_primitives
Removes operations with tiny effects.

    An operation `op` is considered to have a tiny effect if
    `cirq.trace_distance_bound(op) <= atol`.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          atol: Absolute tolerance to determine if an operation `op` is negligible --
                i.e. if `cirq.trace_distance_bound(op) <= atol`.

    Returns:
          Copy of the transformed input circuit.
    