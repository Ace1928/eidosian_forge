from functools import partial
from typing import Sequence, Callable
from collections import OrderedDict
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms.op_transforms import OperationTransformError
from pennylane.transforms import transform
from pennylane.wires import Wires
@partial(transform, is_informative=True)
def _to_zx_transform(tape: QuantumTape, expand_measurements=False) -> (Sequence[QuantumTape], Callable):
    """Private function to convert a PennyLane tape to a `PyZX graph <https://pyzx.readthedocs.io/en/latest/>`_ ."""
    try:
        import pyzx
        from pyzx.circuit.gates import TargetMapper
        from pyzx.graph import Graph
    except ImportError as Error:
        raise ImportError('This feature requires PyZX. It can be installed with: pip install pyzx') from Error
    gate_types = {'PauliX': pyzx.circuit.gates.NOT, 'PauliZ': pyzx.circuit.gates.Z, 'S': pyzx.circuit.gates.S, 'T': pyzx.circuit.gates.T, 'Hadamard': pyzx.circuit.gates.HAD, 'RX': pyzx.circuit.gates.XPhase, 'RZ': pyzx.circuit.gates.ZPhase, 'PhaseShift': pyzx.circuit.gates.ZPhase, 'SWAP': pyzx.circuit.gates.SWAP, 'CNOT': pyzx.circuit.gates.CNOT, 'CZ': pyzx.circuit.gates.CZ, 'CRZ': pyzx.circuit.gates.CRZ, 'CH': pyzx.circuit.gates.CHAD, 'CCZ': pyzx.circuit.gates.CCZ, 'Toffoli': pyzx.circuit.gates.Tofolli}

    def processing_fn(res):
        graph = Graph(None)
        q_mapper = TargetMapper()
        c_mapper = TargetMapper()
        consecutive_wires = Wires(range(len(res[0].wires)))
        consecutive_wires_map = OrderedDict(zip(res[0].wires, consecutive_wires))
        mapped_tapes, fn = qml.map_wires(input=res[0], wire_map=consecutive_wires_map)
        mapped_tape = fn(mapped_tapes)
        inputs = []
        for i in range(len(mapped_tape.wires)):
            vertex = graph.add_vertex(VertexType.BOUNDARY, i, 0)
            inputs.append(vertex)
            q_mapper.set_prev_vertex(i, vertex)
            q_mapper.set_next_row(i, 1)
            q_mapper.set_qubit(i, i)
        stop_crit = qml.BooleanFn(lambda obj: isinstance(obj, Operator) and obj.name in gate_types)
        mapped_tape = qml.tape.tape.expand_tape(mapped_tape, depth=10, stop_at=stop_crit, expand_measurements=expand_measurements)
        expanded_operations = []
        for op in mapped_tape.operations:
            if op.name == 'RY':
                theta = op.data[0]
                decomp = [qml.RX(np.pi / 2, wires=op.wires), qml.RZ(theta + np.pi, wires=op.wires), qml.RX(np.pi / 2, wires=op.wires), qml.RZ(3 * np.pi, wires=op.wires)]
                expanded_operations.extend(decomp)
            else:
                expanded_operations.append(op)
        expanded_tape = QuantumScript(expanded_operations, mapped_tape.measurements)
        _add_operations_to_graph(expanded_tape, graph, gate_types, q_mapper, c_mapper)
        row = max(q_mapper.max_row(), c_mapper.max_row())
        outputs = []
        for mapper in (q_mapper, c_mapper):
            for label in mapper.labels():
                qubit = mapper.to_qubit(label)
                vertex = graph.add_vertex(VertexType.BOUNDARY, qubit, row)
                outputs.append(vertex)
                pre_vertex = mapper.prev_vertex(label)
                graph.add_edge(graph.edge(pre_vertex, vertex))
        graph.set_inputs(tuple(inputs))
        graph.set_outputs(tuple(outputs))
        return graph
    return ([tape], processing_fn)