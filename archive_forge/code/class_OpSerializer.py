from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import numbers
import abc
import numpy as np
import cirq
from cirq.circuits import circuit_operation
from cirq_google.api import v2
from cirq_google.serialization.arg_func_langs import arg_to_proto
class OpSerializer(abc.ABC):
    """Generic supertype for operation serializers.

    Each operation serializer describes how to serialize a specific type of
    Cirq operation to its corresponding proto format. Multiple operation types
    may serialize to the same format.
    """

    @property
    @abc.abstractmethod
    def internal_type(self) -> Type:
        """Returns the type that the operation contains.

        For GateOperations, this is the gate type.
        For CircuitOperations, this is FrozenCircuit.
        """

    @property
    @abc.abstractmethod
    def serialized_id(self) -> str:
        """Returns the string identifier for the resulting serialized object.

        This ID denotes the serialization format this serializer produces. For
        example, one of the common serializers assigns the id 'xy' to XPowGates,
        as they serialize into a format also used by YPowGates.
        """

    @abc.abstractmethod
    def to_proto(self, op, msg=None, *, arg_function_language: Optional[str]='', constants: List[v2.program_pb2.Constant], raw_constants: Dict[Any, int]) -> Optional[v2.program_pb2.CircuitOperation]:
        """Converts op to proto using this serializer.

        If self.can_serialize_operation(op) == false, this should return None.

        Args:
            op: The Cirq operation to be serialized.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The proto-serialized version of `op`. If `msg` was provided, it is
            the returned object.
        """

    @property
    @abc.abstractmethod
    def can_serialize_predicate(self) -> Callable[[cirq.Operation], bool]:
        """The method used to determine if this can serialize an operation.

        Depending on the serializer, additional checks may be required.
        """

    def can_serialize_operation(self, op: cirq.Operation) -> bool:
        """Whether the given operation can be serialized by this serializer."""
        return self.can_serialize_predicate(op)