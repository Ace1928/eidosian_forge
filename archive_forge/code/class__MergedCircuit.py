from collections import defaultdict
import bisect
import dataclasses
from typing import (
from cirq import circuits, ops, protocols
from cirq.circuits.circuit import CIRCUIT_TYPE
@dataclasses.dataclass
class _MergedCircuit:
    """An optimized internal representation of a circuit, tailored for `cirq.merge_operations`

    Attributes:
        qubit_indexes: Mapping from qubits to (sorted) list of moment indexes containing operations
            acting on the qubit.
        mkey_indexes: Mapping from measurement keys to (sorted) list of moment indexes containing
            measurement operations with the same key.
        ckey_indexes: Mapping from measurement keys to (sorted) list of moment indexes containing
            classically controlled operations controlled on the same key.
        ops_by_index: List of circuit moments containing operations. We use a dictionary instead
            of a set to store operations to preserve insertion order.
    """
    qubit_indexes: Dict['cirq.Qid', List[int]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: [-1]))
    mkey_indexes: Dict['cirq.MeasurementKey', List[int]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: [-1]))
    ckey_indexes: Dict['cirq.MeasurementKey', List[int]] = dataclasses.field(default_factory=lambda: defaultdict(lambda: [-1]))
    ops_by_index: List[Dict['cirq.Operation', int]] = dataclasses.field(default_factory=list)

    def append_empty_moment(self) -> None:
        self.ops_by_index.append({})

    def add_op_to_moment(self, moment_index: int, op: 'cirq.Operation') -> None:
        self.ops_by_index[moment_index][op] = 0
        for q in op.qubits:
            if moment_index > self.qubit_indexes[q][-1]:
                self.qubit_indexes[q].append(moment_index)
            else:
                bisect.insort(self.qubit_indexes[q], moment_index)
        for mkey in protocols.measurement_key_objs(op):
            bisect.insort(self.mkey_indexes[mkey], moment_index)
        for ckey in protocols.control_keys(op):
            bisect.insort(self.ckey_indexes[ckey], moment_index)

    def remove_op_from_moment(self, moment_index: int, op: 'cirq.Operation') -> None:
        self.ops_by_index[moment_index].pop(op)
        for q in op.qubits:
            if self.qubit_indexes[q][-1] == moment_index:
                self.qubit_indexes[q].pop()
            else:
                self.qubit_indexes[q].remove(moment_index)
        for mkey in protocols.measurement_key_objs(op):
            self.mkey_indexes[mkey].remove(moment_index)
        for ckey in protocols.control_keys(op):
            self.ckey_indexes[ckey].remove(moment_index)

    def get_mergeable_ops(self, op: 'cirq.Operation', op_qs: Set['cirq.Qid']) -> Tuple[int, List['cirq.Operation']]:
        idx = max([self.qubit_indexes[q][-1] for q in op_qs], default=-1)
        idx = max([idx] + [self.mkey_indexes[ckey][-1] for ckey in protocols.control_keys(op)])
        idx = max([idx] + [self.ckey_indexes[mkey][-1] for mkey in protocols.measurement_key_objs(op)])
        if idx == -1:
            return (idx, [])
        return (idx, [left_op for left_op in self.ops_by_index[idx] if not op_qs.isdisjoint(left_op.qubits)])

    def get_cirq_circuit(self) -> 'cirq.Circuit':
        return circuits.Circuit((circuits.Moment(m.keys()) for m in self.ops_by_index))