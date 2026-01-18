from typing import Any, Dict, List, Optional, Set, Sequence, Tuple, TYPE_CHECKING
import itertools
import networkx as nx
from cirq import circuits, ops, protocols
from cirq.transformers import transformer_api, transformer_primitives
from cirq.transformers.routing import mapping_manager, line_initial_mapper
@classmethod
def _get_one_and_two_qubit_ops_as_timesteps(cls, circuit: 'cirq.AbstractCircuit') -> Tuple[List[List['cirq.Operation']], List[List['cirq.Operation']]]:
    """Gets the single and two qubit operations of the circuit factored into timesteps.

        The i'th entry in the nested two-qubit and single-qubit ops correspond to the two-qubit
        gates and single-qubit gates of the i'th timesteps respectively. When constructing the
        output routed circuit, single-qubit operations are inserted before two-qubit operations.

        Raises:
            ValueError: if circuit has intermediate measurements that act on three or more
                        qubits with a custom key.
        """
    two_qubit_circuit = circuits.Circuit()
    single_qubit_ops: List[List[cirq.Operation]] = []
    for i, moment in enumerate(circuit):
        for op in moment:
            timestep = two_qubit_circuit.earliest_available_moment(op)
            single_qubit_ops.extend(([] for _ in range(timestep + 1 - len(single_qubit_ops))))
            two_qubit_circuit.append((circuits.Moment() for _ in range(timestep + 1 - len(two_qubit_circuit))))
            if protocols.num_qubits(op) > 2 and protocols.is_measurement(op):
                key = op.gate.key
                default_key = ops.measure(op.qubits).gate.key
                if len(circuit.moments) == i + 1:
                    single_qubit_ops[timestep].append(op)
                elif key in ('', default_key):
                    single_qubit_ops[timestep].extend((ops.measure(qubit) for qubit in op.qubits))
                else:
                    raise ValueError('Intermediate measurements on three or more qubits with a custom key are not supported')
            elif protocols.num_qubits(op) == 2:
                two_qubit_circuit[timestep] = two_qubit_circuit[timestep].with_operation(op)
            else:
                single_qubit_ops[timestep].append(op)
    two_qubit_ops = [list(m) for m in two_qubit_circuit]
    return (two_qubit_ops, single_qubit_ops)