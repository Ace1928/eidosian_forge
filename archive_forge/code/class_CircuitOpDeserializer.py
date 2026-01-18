from typing import Any, List
import abc
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.serialization import arg_func_langs
class CircuitOpDeserializer(OpDeserializer):
    """Describes how to serialize CircuitOperations."""

    @property
    def serialized_id(self):
        return 'circuit'

    def from_proto(self, proto: v2.program_pb2.CircuitOperation, *, arg_function_language: str='', constants: List[v2.program_pb2.Constant], deserialized_constants: List[Any]) -> cirq.CircuitOperation:
        """Turns a cirq.google.api.v2.CircuitOperation proto into a CircuitOperation.

        Args:
            proto: The proto object to be deserialized.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`. This list should already have been
                parsed to produce 'deserialized_constants'.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized CircuitOperation represented by `proto`.

        Raises:
            ValueError: If the circuit operatio proto cannot be deserialied because it is malformed.
        """
        if len(deserialized_constants) <= proto.circuit_constant_index:
            raise ValueError(f'Constant index {proto.circuit_constant_index} in CircuitOperation does not appear in the deserialized_constants list (length {len(deserialized_constants)}).')
        circuit = deserialized_constants[proto.circuit_constant_index]
        if not isinstance(circuit, cirq.FrozenCircuit):
            raise ValueError(f'Constant at index {proto.circuit_constant_index} was expected to be a circuit, but it has type {type(circuit)} in the deserialized_constants list.')
        which_rep_spec = proto.repetition_specification.WhichOneof('repetition_value')
        if which_rep_spec == 'repetition_count':
            rep_ids = None
            repetitions = proto.repetition_specification.repetition_count
        elif which_rep_spec == 'repetition_ids':
            rep_ids = proto.repetition_specification.repetition_ids.ids
            repetitions = len(rep_ids)
        else:
            rep_ids = None
            repetitions = 1
        qubit_map = {v2.qubit_from_proto_id(entry.key.id): v2.qubit_from_proto_id(entry.value.id) for entry in proto.qubit_map.entries}
        measurement_key_map = {entry.key.string_key: entry.value.string_key for entry in proto.measurement_key_map.entries}
        arg_map = {arg_func_langs.arg_from_proto(entry.key, arg_function_language=arg_function_language): arg_func_langs.arg_from_proto(entry.value, arg_function_language=arg_function_language) for entry in proto.arg_map.entries}
        for arg in arg_map.keys():
            if not isinstance(arg, (str, sympy.Symbol)):
                raise ValueError(f'Invalid key parameter type in deserialized CircuitOperation. Expected str or sympy.Symbol, found {type(arg)}.\nFull arg: {arg}')
        for arg in arg_map.values():
            if not isinstance(arg, (str, sympy.Symbol, float, int)):
                raise ValueError(f'Invalid value parameter type in deserialized CircuitOperation. Expected str, sympy.Symbol, or number; found {type(arg)}.\nFull arg: {arg}')
        return cirq.CircuitOperation(circuit, repetitions, qubit_map, measurement_key_map, arg_map, rep_ids)