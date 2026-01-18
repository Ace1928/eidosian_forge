from typing import Any, Dict, List, Optional
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.ops import PhysicalZTag, InternalGate
from cirq_google.ops.calibration_tag import CalibrationTag
from cirq_google.serialization import serializer, op_deserializer, op_serializer, arg_func_langs
def _serialize_gate_op(self, op: cirq.Operation, msg: v2.program_pb2.Operation, *, constants: List[v2.program_pb2.Constant], raw_constants: Dict[Any, int], arg_function_language: Optional[str]='') -> v2.program_pb2.Operation:
    """Serialize an Operation to cirq_google.api.v2.Operation proto.

        Args:
            op: The operation to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The cirq.google.api.v2.Operation proto.

        Raises:
            ValueError: If the operation cannot be serialized.
        """
    gate = op.gate
    if isinstance(gate, InternalGate):
        arg_func_langs.internal_gate_arg_to_proto(gate, out=msg.internalgate)
    elif isinstance(gate, cirq.XPowGate):
        arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.xpowgate.exponent, arg_function_language=arg_function_language)
    elif isinstance(gate, cirq.YPowGate):
        arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.ypowgate.exponent, arg_function_language=arg_function_language)
    elif isinstance(gate, cirq.ZPowGate):
        arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.zpowgate.exponent, arg_function_language=arg_function_language)
        if any((isinstance(tag, PhysicalZTag) for tag in op.tags)):
            msg.zpowgate.is_physical_z = True
    elif isinstance(gate, cirq.PhasedXPowGate):
        arg_func_langs.float_arg_to_proto(gate.phase_exponent, out=msg.phasedxpowgate.phase_exponent, arg_function_language=arg_function_language)
        arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.phasedxpowgate.exponent, arg_function_language=arg_function_language)
    elif isinstance(gate, cirq.PhasedXZGate):
        arg_func_langs.float_arg_to_proto(gate.x_exponent, out=msg.phasedxzgate.x_exponent, arg_function_language=arg_function_language)
        arg_func_langs.float_arg_to_proto(gate.z_exponent, out=msg.phasedxzgate.z_exponent, arg_function_language=arg_function_language)
        arg_func_langs.float_arg_to_proto(gate.axis_phase_exponent, out=msg.phasedxzgate.axis_phase_exponent, arg_function_language=arg_function_language)
    elif isinstance(gate, cirq.CZPowGate):
        arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.czpowgate.exponent, arg_function_language=arg_function_language)
    elif isinstance(gate, cirq.ISwapPowGate):
        arg_func_langs.float_arg_to_proto(gate.exponent, out=msg.iswappowgate.exponent, arg_function_language=arg_function_language)
    elif isinstance(gate, cirq.FSimGate):
        arg_func_langs.float_arg_to_proto(gate.theta, out=msg.fsimgate.theta, arg_function_language=arg_function_language)
        arg_func_langs.float_arg_to_proto(gate.phi, out=msg.fsimgate.phi, arg_function_language=arg_function_language)
    elif isinstance(gate, cirq.MeasurementGate):
        arg_func_langs.arg_to_proto(gate.key, out=msg.measurementgate.key, arg_function_language=arg_function_language)
        arg_func_langs.arg_to_proto(gate.invert_mask, out=msg.measurementgate.invert_mask, arg_function_language=arg_function_language)
    elif isinstance(gate, cirq.WaitGate):
        arg_func_langs.float_arg_to_proto(gate.duration.total_nanos(), out=msg.waitgate.duration_nanos, arg_function_language=arg_function_language)
    else:
        raise ValueError(f'Cannot serialize op {op!r} of type {type(gate)}')
    for qubit in op.qubits:
        if qubit not in raw_constants:
            constants.append(v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id=v2.qubit_to_proto_id(qubit))))
            raw_constants[qubit] = len(constants) - 1
        msg.qubit_constant_index.append(raw_constants[qubit])
    for tag in op.tags:
        if isinstance(tag, CalibrationTag):
            constant = v2.program_pb2.Constant()
            constant.string_value = tag.token
            if tag.token in raw_constants:
                msg.token_constant_index = raw_constants[tag.token]
            else:
                msg.token_constant_index = len(constants)
                constants.append(constant)
                if raw_constants is not None:
                    raw_constants[tag.token] = msg.token_constant_index
    return msg