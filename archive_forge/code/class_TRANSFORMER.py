import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
class TRANSFORMER(Protocol):
    """Protocol class defining the Transformer API for circuit transformers in Cirq.

    Any callable that satisfies the `cirq.TRANSFORMER` contract, i.e. takes a `cirq.AbstractCircuit`
    and `cirq.TransformerContext` and returns a transformed `cirq.AbstractCircuit`, is a valid
    transformer in Cirq.

    Note that transformers can also accept additional arguments as `**kwargs`, with default values
    specified for each keyword argument. A transformer could be a function, for example:

    >>> def convert_to_cz(
    ...     circuit: cirq.AbstractCircuit,
    ...     *,
    ...     context: 'Optional[cirq.TransformerContext]' = None,
    ...     atol: float = 1e-8,
    ... ) -> cirq.Circuit:
    ...     ...

    Or it could be a class that implements `__call__` with the same API, for example:

    >>> class ConvertToSqrtISwaps:
    ...     def __init__(self):
    ...         ...
    ...     def __call__(
    ...         self,
    ...         circuit: cirq.AbstractCircuit,
    ...         *,
    ...         context: 'Optional[cirq.TransformerContext]' = None,
    ...      ) -> cirq.AbstractCircuit:
    ...         ...
    """

    def __call__(self, circuit: 'cirq.AbstractCircuit', *, context: Optional[TransformerContext]=None) -> 'cirq.AbstractCircuit':
        ...