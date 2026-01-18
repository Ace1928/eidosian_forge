from typing import Any, List
import abc
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.serialization import arg_func_langs
class OpDeserializer(abc.ABC):
    """Generic supertype for operation deserializers.

    Each operation deserializer describes how to deserialize operation protos
    with a particular `serialized_id` to a specific type of Cirq operation.
    """

    @property
    @abc.abstractmethod
    def serialized_id(self) -> str:
        """Returns the string identifier for the accepted serialized objects.

        This ID denotes the serialization format this deserializer consumes. For
        example, one of the common deserializers converts objects with the id
        'xy' into PhasedXPowGates.
        """

    @abc.abstractmethod
    def from_proto(self, proto, *, arg_function_language: str='', constants: List[v2.program_pb2.Constant], deserialized_constants: List[Any]) -> cirq.Operation:
        """Converts a proto-formatted operation into a Cirq operation.

        Args:
            proto: The proto object to be deserialized.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized operation represented by `proto`.
        """