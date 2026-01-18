from typing import Any, Callable, Dict, Generic, Iterator, TypeVar, cast, TYPE_CHECKING
import functools
import networkx
from cirq import ops
from cirq.circuits import circuit
class CircuitDag(networkx.DiGraph):
    """A representation of a Circuit as a directed acyclic graph.

    Nodes of the graph are instances of Unique containing each operation of a
    circuit.

    Edges of the graph are tuples of nodes.  Each edge specifies a required
    application order between two operations.  The first must be applied before
    the second.

    The graph is maximalist (transitive completion).
    """
    disjoint_qubits = staticmethod(_disjoint_qubits)

    def __init__(self, can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool]=_disjoint_qubits, incoming_graph_data: Any=None) -> None:
        """Initializes a CircuitDag.

        Args:
            can_reorder: A predicate that determines if two operations may be
                reordered.  Graph edges are created for pairs of operations
                where this returns False.

                The default predicate allows reordering only when the operations
                don't share common qubits.
            incoming_graph_data: Data in initialize the graph.  This can be any
                value supported by networkx.DiGraph() e.g. an edge list or
                another graph.
            device: Hardware that the circuit should be able to run on.
        """
        super().__init__(incoming_graph_data)
        self.can_reorder = can_reorder

    @staticmethod
    def make_node(op: 'cirq.Operation') -> Unique:
        return Unique(op)

    @staticmethod
    def from_circuit(circuit: circuit.Circuit, can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool]=_disjoint_qubits) -> 'CircuitDag':
        return CircuitDag.from_ops(circuit.all_operations(), can_reorder=can_reorder)

    @staticmethod
    def from_ops(*operations: 'cirq.OP_TREE', can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool]=_disjoint_qubits) -> 'CircuitDag':
        dag = CircuitDag(can_reorder=can_reorder)
        for op in ops.flatten_op_tree(operations):
            dag.append(cast(ops.Operation, op))
        return dag

    def append(self, op: 'cirq.Operation') -> None:
        new_node = self.make_node(op)
        for node in list(self.nodes()):
            if not self.can_reorder(node.val, op):
                self.add_edge(node, new_node)
                for pred in self.pred[node]:
                    self.add_edge(pred, new_node)
        self.add_node(new_node)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        g1 = self.copy()
        g2 = other.copy()
        for node, attr in g1.nodes(data=True):
            attr['val'] = node.val
        for node, attr in g2.nodes(data=True):
            attr['val'] = node.val

        def node_match(attr1: Dict[Any, Any], attr2: Dict[Any, Any]) -> bool:
            return attr1['val'] == attr2['val']
        return networkx.is_isomorphic(g1, g2, node_match=node_match)

    def __ne__(self, other):
        return not self == other
    __hash__ = None

    def ordered_nodes(self) -> Iterator[Unique['cirq.Operation']]:
        if not self.nodes():
            return
        g = self.copy()

        def get_root_node(some_node: Unique['cirq.Operation']) -> Unique['cirq.Operation']:
            pred = g.pred
            while pred[some_node]:
                some_node = next(iter(pred[some_node]))
            return some_node

        def get_first_node() -> Unique['cirq.Operation']:
            return get_root_node(next(iter(g.nodes())))

        def get_next_node(succ: networkx.classes.coreviews.AtlasView) -> Unique['cirq.Operation']:
            if succ:
                return get_root_node(next(iter(succ)))
            return get_first_node()
        node = get_first_node()
        while True:
            yield node
            succ = g.succ[node]
            g.remove_node(node)
            if not g.nodes():
                return
            node = get_next_node(succ)

    def all_operations(self) -> Iterator['cirq.Operation']:
        return (node.val for node in self.ordered_nodes())

    def all_qubits(self):
        return frozenset((q for node in self.nodes for q in node.val.qubits))

    def to_circuit(self) -> circuit.Circuit:
        return circuit.Circuit(self.all_operations(), strategy=circuit.InsertStrategy.EARLIEST)

    def findall_nodes_until_blocked(self, is_blocker: Callable[['cirq.Operation'], bool]) -> Iterator[Unique['cirq.Operation']]:
        """Finds all nodes before blocking ones.

        Args:
            is_blocker: The predicate that indicates whether or not an
            operation is blocking.
        """
        remaining_dag = self.copy()
        for node in self.ordered_nodes():
            if node not in remaining_dag:
                continue
            if is_blocker(node.val):
                successors = list(remaining_dag.succ[node])
                remaining_dag.remove_nodes_from(successors)
                remaining_dag.remove_node(node)
                continue
            yield node