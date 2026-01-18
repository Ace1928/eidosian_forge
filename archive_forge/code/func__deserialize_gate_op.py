from typing import Any, Dict, List, Optional
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.ops import PhysicalZTag, InternalGate
from cirq_google.ops.calibration_tag import CalibrationTag
from cirq_google.serialization import serializer, op_deserializer, op_serializer, arg_func_langs
def _deserialize_gate_op(self, operation_proto: v2.program_pb2.Operation, *, arg_function_language: str='', constants: Optional[List[v2.program_pb2.Constant]]=None, deserialized_constants: Optional[List[Any]]=None) -> cirq.Operation:
    """Deserialize an Operation from a cirq_google.api.v2.Operation.

        Args:
            operation_proto: A dictionary representing a
                cirq.google.api.v2.Operation proto.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.
                cirq_google.api.v2.Operation proto.

        Returns:
            The deserialized Operation.

        Raises:
            ValueError: If the operation cannot be deserialized.
        """
    if deserialized_constants is not None:
        qubits = [deserialized_constants[q] for q in operation_proto.qubit_constant_index]
    else:
        qubits = []
    for q in operation_proto.qubits:
        qubits.append(v2.qubit_from_proto_id(q.id))
    which_gate_type = operation_proto.WhichOneof('gate_value')
    if which_gate_type == 'xpowgate':
        op = cirq.XPowGate(exponent=arg_func_langs.float_arg_from_proto(operation_proto.xpowgate.exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0)(*qubits)
    elif which_gate_type == 'ypowgate':
        op = cirq.YPowGate(exponent=arg_func_langs.float_arg_from_proto(operation_proto.ypowgate.exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0)(*qubits)
    elif which_gate_type == 'zpowgate':
        op = cirq.ZPowGate(exponent=arg_func_langs.float_arg_from_proto(operation_proto.zpowgate.exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0)(*qubits)
        if operation_proto.zpowgate.is_physical_z:
            op = op.with_tags(PhysicalZTag())
    elif which_gate_type == 'phasedxpowgate':
        exponent = arg_func_langs.float_arg_from_proto(operation_proto.phasedxpowgate.exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0
        phase_exponent = arg_func_langs.float_arg_from_proto(operation_proto.phasedxpowgate.phase_exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0
        op = cirq.PhasedXPowGate(exponent=exponent, phase_exponent=phase_exponent)(*qubits)
    elif which_gate_type == 'phasedxzgate':
        x_exponent = arg_func_langs.float_arg_from_proto(operation_proto.phasedxzgate.x_exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0
        z_exponent = arg_func_langs.float_arg_from_proto(operation_proto.phasedxzgate.z_exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0
        axis_phase_exponent = arg_func_langs.float_arg_from_proto(operation_proto.phasedxzgate.axis_phase_exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0
        op = cirq.PhasedXZGate(x_exponent=x_exponent, z_exponent=z_exponent, axis_phase_exponent=axis_phase_exponent)(*qubits)
    elif which_gate_type == 'czpowgate':
        op = cirq.CZPowGate(exponent=arg_func_langs.float_arg_from_proto(operation_proto.czpowgate.exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0)(*qubits)
    elif which_gate_type == 'iswappowgate':
        op = cirq.ISwapPowGate(exponent=arg_func_langs.float_arg_from_proto(operation_proto.iswappowgate.exponent, arg_function_language=arg_function_language, required_arg_name=None) or 0.0)(*qubits)
    elif which_gate_type == 'fsimgate':
        theta = arg_func_langs.float_arg_from_proto(operation_proto.fsimgate.theta, arg_function_language=arg_function_language, required_arg_name=None)
        phi = arg_func_langs.float_arg_from_proto(operation_proto.fsimgate.phi, arg_function_language=arg_function_language, required_arg_name=None)
        if isinstance(theta, (int, float, sympy.Basic)) and isinstance(phi, (int, float, sympy.Basic)):
            op = cirq.FSimGate(theta=theta, phi=phi)(*qubits)
        else:
            raise ValueError('theta and phi must be specified for FSimGate')
    elif which_gate_type == 'measurementgate':
        key = arg_func_langs.arg_from_proto(operation_proto.measurementgate.key, arg_function_language=arg_function_language, required_arg_name=None)
        parsed_invert_mask = arg_func_langs.arg_from_proto(operation_proto.measurementgate.invert_mask, arg_function_language=arg_function_language, required_arg_name=None)
        if (isinstance(parsed_invert_mask, list) or parsed_invert_mask is None) and isinstance(key, str):
            invert_mask: tuple[bool, ...] = ()
            if parsed_invert_mask is not None:
                invert_mask = tuple((bool(x) for x in parsed_invert_mask))
            op = cirq.MeasurementGate(num_qubits=len(qubits), key=key, invert_mask=invert_mask)(*qubits)
        else:
            raise ValueError(f'Incorrect types for measurement gate {parsed_invert_mask} {key}')
    elif which_gate_type == 'waitgate':
        total_nanos = arg_func_langs.float_arg_from_proto(operation_proto.waitgate.duration_nanos, arg_function_language=arg_function_language, required_arg_name=None)
        op = cirq.WaitGate(duration=cirq.Duration(nanos=total_nanos or 0.0))(*qubits)
    elif which_gate_type == 'internalgate':
        op = arg_func_langs.internal_gate_from_proto(operation_proto.internalgate, arg_function_language=arg_function_language)(*qubits)
    else:
        raise ValueError(f'Unsupported serialized gate with type "{which_gate_type}".\n\noperation_proto:\n{operation_proto}')
    which = operation_proto.WhichOneof('token')
    if which == 'token_constant_index':
        if not constants:
            raise ValueError(f'Proto has references to constants table but none was passed in, value ={operation_proto}')
        op = op.with_tags(CalibrationTag(constants[operation_proto.token_constant_index].string_value))
    elif which == 'token_value':
        op = op.with_tags(CalibrationTag(operation_proto.token_value))
    return op