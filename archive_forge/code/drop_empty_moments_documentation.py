from typing import Optional, TYPE_CHECKING
from cirq.transformers import transformer_api, transformer_primitives
Removes empty moments from a circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
          Copy of the transformed input circuit.
    