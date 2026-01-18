from typing import Any, Dict, List, Optional
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.ops import PhysicalZTag, InternalGate
from cirq_google.ops.calibration_tag import CalibrationTag
from cirq_google.serialization import serializer, op_deserializer, op_serializer, arg_func_langs
def _deserialize_circuit_op(self, operation_proto: v2.program_pb2.CircuitOperation, *, arg_function_language: str='', constants: List[v2.program_pb2.Constant], deserialized_constants: List[Any]) -> cirq.CircuitOperation:
    """Deserialize a CircuitOperation from a
            cirq.google.api.v2.CircuitOperation.

        Args:
            operation_proto: A dictionary representing a
                cirq.google.api.v2.CircuitOperation proto.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized CircuitOperation.
        """
    return op_deserializer.CircuitOpDeserializer().from_proto(operation_proto, arg_function_language=arg_function_language, constants=constants, deserialized_constants=deserialized_constants)