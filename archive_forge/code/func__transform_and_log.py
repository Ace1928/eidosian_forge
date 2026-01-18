import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
def _transform_and_log(add_deep_support: bool, func: TRANSFORMER, transformer_name: str, circuit: 'cirq.AbstractCircuit', extracted_context: Optional[TransformerContext], **kwargs) -> 'cirq.AbstractCircuit':
    """Helper to log initial and final circuits before and after calling the transformer."""
    if extracted_context:
        extracted_context.logger.register_initial(circuit, transformer_name)
    transformed_circuit = _run_transformer_on_circuit(add_deep_support, func, circuit, extracted_context, **kwargs)
    if extracted_context:
        extracted_context.logger.register_final(transformed_circuit, transformer_name)
    return transformed_circuit