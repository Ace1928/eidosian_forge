import string
from typing import Callable, Dict, Set, Tuple, Union, Any, Optional, List, cast
import numpy as np
import cirq
import cirq_rigetti
from cirq import protocols, value, ops
class QuilOutput:
    """An object for passing operations and qubits then outputting them to
    QUIL format. The string representation returns the QUIL output for the
    circuit.
    """

    def __init__(self, operations: 'cirq.OP_TREE', qubits: Tuple['cirq.Qid', ...]) -> None:
        """Inits QuilOutput.

        Args:
            operations: A list or tuple of `cirq.OP_TREE` arguments.
            qubits: The qubits used in the operations.
        """
        self.qubits = qubits
        self.operations = tuple(cirq.ops.flatten_to_ops(operations))
        self.measurements = tuple((op for op in self.operations if isinstance(op.gate, ops.MeasurementGate)))
        self.qubit_id_map = self._generate_qubit_ids()
        self.measurement_id_map = self._generate_measurement_ids()
        self.formatter = cirq_rigetti.quil_output.QuilFormatter(qubit_id_map=self.qubit_id_map, measurement_id_map=self.measurement_id_map)

    def _generate_qubit_ids(self) -> Dict['cirq.Qid', str]:
        return {qubit: str(i) for i, qubit in enumerate(self.qubits)}

    def _generate_measurement_ids(self) -> Dict[str, str]:
        index = 0
        measurement_id_map: Dict[str, str] = {}
        for op in self.operations:
            if isinstance(op.gate, ops.MeasurementGate):
                key = protocols.measurement_key_name(op)
                if key in measurement_id_map:
                    continue
                measurement_id_map[key] = f'm{index}'
                index += 1
        return measurement_id_map

    def save_to_file(self, path: Union[str, bytes, int]) -> None:
        """Write QUIL output to a file specified by path."""
        with open(path, 'w') as f:
            f.write(str(self))

    def __str__(self) -> str:
        output = []
        self._write_quil(lambda s: output.append(s))
        return self.rename_defgates(''.join(output))

    def _op_to_maybe_quil(self, op: cirq.Operation) -> Optional[str]:
        for gate_type in SUPPORTED_GATES.keys():
            if isinstance(op.gate, gate_type):
                quil: Callable[[cirq.Operation, QuilFormatter], Optional[str]] = SUPPORTED_GATES[gate_type]
                return quil(op, self.formatter)
        return None

    def _op_to_quil(self, op: cirq.Operation) -> str:
        quil_str = self._op_to_maybe_quil(op)
        if not quil_str:
            raise ValueError("Can't convert Operation to string")
        return quil_str

    def _write_quil(self, output_func: Callable[[str], None]) -> None:
        output_func('# Created using Cirq.\n\n')
        if len(self.measurements) > 0:
            measurements_declared: Set[str] = set()
            for m in self.measurements:
                key = protocols.measurement_key_name(m)
                if key in measurements_declared:
                    continue
                measurements_declared.add(key)
                output_func(f'DECLARE {self.measurement_id_map[key]} BIT[{len(m.qubits)}]\n')
            output_func('\n')

        def keep(op: 'cirq.Operation') -> bool:
            if isinstance(op.gate, tuple(SUPPORTED_GATES.keys())):
                if not self._op_to_maybe_quil(op):
                    return False
                return True
            return False

        def fallback(op):
            if len(op.qubits) not in [1, 2]:
                return NotImplemented
            mat = protocols.unitary(op, None)
            if mat is None:
                return NotImplemented
            if len(op.qubits) == 1:
                return QuilOneQubitGate(mat).on(*op.qubits)
            return QuilTwoQubitGate(mat).on(*op.qubits)

        def on_stuck(bad_op):
            return ValueError(f'Cannot output operation as QUIL: {bad_op!r}')
        for main_op in self.operations:
            decomposed = protocols.decompose(main_op, keep=keep, fallback_decomposer=fallback, on_stuck_raise=on_stuck)
            for decomposed_op in decomposed:
                output_func(self._op_to_quil(decomposed_op))

    def rename_defgates(self, output: str) -> str:
        """A function for renaming the DEFGATEs within the QUIL output. This
        utilizes a second pass to find each DEFGATE and rename it based on
        a counter.
        """
        result = output
        defString = 'DEFGATE'
        nameString = 'USERGATE'
        defIdx = 0
        nameIdx = 0
        gateNum = 0
        i = 0
        while i < len(output):
            if result[i] == defString[defIdx]:
                defIdx += 1
            else:
                defIdx = 0
            if result[i] == nameString[nameIdx]:
                nameIdx += 1
            else:
                nameIdx = 0
            if defIdx == len(defString):
                gateNum += 1
                defIdx = 0
            if nameIdx == len(nameString):
                result = result[:i + 1] + str(gateNum) + result[i + 1:]
                nameIdx = 0
                i += 1
            i += 1
        return result