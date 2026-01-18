import dataclasses
from typing import Optional, TYPE_CHECKING
from cirq import circuits, ops
from cirq.transformers import transformer_api
Align gates to the right of the circuit.

    Note that tagged operations with tag in `context.tags_to_ignore` will continue to stay in their
    original position and will not be aligned.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
          Copy of the transformed input circuit.
    