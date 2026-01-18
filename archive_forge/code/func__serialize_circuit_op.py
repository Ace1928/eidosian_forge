from typing import Any, Dict, List, Optional
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.ops import PhysicalZTag, InternalGate
from cirq_google.ops.calibration_tag import CalibrationTag
from cirq_google.serialization import serializer, op_deserializer, op_serializer, arg_func_langs
def _serialize_circuit_op(self, op: cirq.CircuitOperation, msg: Optional[v2.program_pb2.CircuitOperation]=None, *, arg_function_language: Optional[str]='', constants: Optional[List[v2.program_pb2.Constant]]=None, raw_constants: Optional[Dict[Any, int]]=None) -> v2.program_pb2.CircuitOperation:
    """Serialize a CircuitOperation to cirq.google.api.v2.CircuitOperation proto.

        Args:
            op: The circuit operation to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The cirq.google.api.v2.CircuitOperation proto.

        Raises:
            ValueError: If `constant` or `raw_constants` are not specified.
        """
    circuit = op.circuit
    if constants is None or raw_constants is None:
        raise ValueError('CircuitOp serialization requires a constants list and a corresponding map of pre-serialization values to indices (raw_constants).')
    serializer = op_serializer.CircuitOpSerializer()
    if circuit not in raw_constants:
        subcircuit_msg = v2.program_pb2.Circuit()
        self._serialize_circuit(circuit, subcircuit_msg, arg_function_language=arg_function_language, constants=constants, raw_constants=raw_constants)
        constants.append(v2.program_pb2.Constant(circuit_value=subcircuit_msg))
        raw_constants[circuit] = len(constants) - 1
    return serializer.to_proto(op, msg, arg_function_language=arg_function_language, constants=constants, raw_constants=raw_constants)