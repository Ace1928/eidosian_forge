from typing import Deque, Dict, List, Set, Tuple, TYPE_CHECKING
from collections import deque
import networkx as nx
from cirq.transformers.routing import initial_mapper
from cirq import protocols, value
def _make_circuit_graph(self, circuit: 'cirq.AbstractCircuit') -> Tuple[List[Deque['cirq.Qid']], Dict['cirq.Qid', 'cirq.Qid']]:
    """Creates a (potentially incomplete) qubit connectivity graph of the circuit.

        Iterates over moments in the circuit from left to right and adds edges between logical
        qubits if the logical qubit pair l1 and l2
            (1) have degree < 2,
            (2) are involved in a 2-qubit operation in the current moment, and
            (3) adding such an edge will not produce a cycle in the graph.

        Args:
            circuit: the input circuit with logical qubits

        Returns:
            The (potentially incomplete) qubit connectivity graph of the circuit, which is
                guaranteed to be a forest of line graphs.
        """
    circuit_graph: List[Deque['cirq.Qid']] = [deque([q]) for q in sorted(circuit.all_qubits())]
    component_id: Dict['cirq.Qid', int] = {q[0]: i for i, q in enumerate(circuit_graph)}
    partners: Dict['cirq.Qid', 'cirq.Qid'] = {}

    def degree_lt_two(q: 'cirq.Qid'):
        return any((circuit_graph[component_id[q]][i] == q for i in [-1, 0]))
    for op in circuit.all_operations():
        if protocols.num_qubits(op) != 2:
            continue
        q0, q1 = op.qubits
        c0, c1 = (component_id[q0], component_id[q1])
        partners[q0] = partners[q0] if q0 in partners else q1
        partners[q1] = partners[q1] if q1 in partners else q0
        if not (degree_lt_two(q0) and degree_lt_two(q1) and (c0 != c1)):
            continue
        if len(circuit_graph[c0]) < len(circuit_graph[c1]):
            c0, c1, q0, q1 = (c1, c0, q1, q0)
        c1_order = reversed(circuit_graph[c1]) if circuit_graph[c1][-1] == q1 else iter(circuit_graph[c1])
        for q in c1_order:
            if circuit_graph[c0][0] == q0:
                circuit_graph[c0].appendleft(q)
            else:
                circuit_graph[c0].append(q)
            component_id[q] = c0
    graph = sorted([circuit_graph[c] for c in set(component_id.values())], key=len, reverse=True)
    return (graph, partners)