from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
class BlockCollector:
    """This class implements various strategies of dividing a DAG (direct acyclic graph)
    into blocks of nodes that satisfy certain criteria. It works both with the
    :class:`~qiskit.dagcircuit.DAGCircuit` and
    :class:`~qiskit.dagcircuit.DAGDependency` representations of a DAG, where
    DagDependency takes into account commutativity between nodes.

    Collecting nodes from DAGDependency generally leads to more optimal results, but is
    slower, as it requires to construct a DAGDependency beforehand. Thus, DAGCircuit should
    be used with lower transpiler settings, and DAGDependency should be used with higher
    transpiler settings.

    In general, there are multiple ways to collect maximal blocks. The approaches used
    here are of the form 'starting from the input nodes of a DAG, greedily collect
    the largest block of nodes that match certain criteria'. For additional details,
    see https://github.com/Qiskit/qiskit-terra/issues/5775.
    """

    def __init__(self, dag):
        """
        Args:
            dag (Union[DAGCircuit, DAGDependency]): The input DAG.

        Raises:
            DAGCircuitError: the input object is not a DAG.
        """
        self.dag = dag
        self._pending_nodes = None
        self._in_degree = None
        self._collect_from_back = False
        if isinstance(dag, DAGCircuit):
            self.is_dag_dependency = False
        elif isinstance(dag, DAGDependency):
            self.is_dag_dependency = True
        else:
            raise DAGCircuitError('not a DAG.')

    def _setup_in_degrees(self):
        """For an efficient implementation, for every node we keep the number of its
        unprocessed immediate predecessors (called ``_in_degree``). This ``_in_degree``
        is set up at the start and updated throughout the algorithm.
        A node is leaf (or input) node iff its ``_in_degree`` is 0.
        When a node is (marked as) collected, the ``_in_degree`` of each of its immediate
        successor is updated by subtracting 1.
        Additionally, ``_pending_nodes`` explicitly keeps the list of nodes whose
        ``_in_degree`` is 0.
        """
        self._pending_nodes = []
        self._in_degree = {}
        for node in self._op_nodes():
            deg = len(self._direct_preds(node))
            self._in_degree[node] = deg
            if deg == 0:
                self._pending_nodes.append(node)

    def _op_nodes(self):
        """Returns DAG nodes."""
        if not self.is_dag_dependency:
            return self.dag.op_nodes()
        else:
            return self.dag.get_nodes()

    def _direct_preds(self, node):
        """Returns direct predecessors of a node. This function takes into account the
        direction of collecting blocks, that is node's predecessors when collecting
        backwards are the direct successors of a node in the DAG.
        """
        if not self.is_dag_dependency:
            if self._collect_from_back:
                return [pred for pred in self.dag.successors(node) if isinstance(pred, DAGOpNode)]
            else:
                return [pred for pred in self.dag.predecessors(node) if isinstance(pred, DAGOpNode)]
        elif self._collect_from_back:
            return [self.dag.get_node(pred_id) for pred_id in self.dag.direct_successors(node.node_id)]
        else:
            return [self.dag.get_node(pred_id) for pred_id in self.dag.direct_predecessors(node.node_id)]

    def _direct_succs(self, node):
        """Returns direct successors of a node. This function takes into account the
        direction of collecting blocks, that is node's successors when collecting
        backwards are the direct predecessors of a node in the DAG.
        """
        if not self.is_dag_dependency:
            if self._collect_from_back:
                return [succ for succ in self.dag.predecessors(node) if isinstance(succ, DAGOpNode)]
            else:
                return [succ for succ in self.dag.successors(node) if isinstance(succ, DAGOpNode)]
        elif self._collect_from_back:
            return [self.dag.get_node(succ_id) for succ_id in self.dag.direct_predecessors(node.node_id)]
        else:
            return [self.dag.get_node(succ_id) for succ_id in self.dag.direct_successors(node.node_id)]

    def _have_uncollected_nodes(self):
        """Returns whether there are uncollected (pending) nodes"""
        return len(self._pending_nodes) > 0

    def collect_matching_block(self, filter_fn):
        """Iteratively collects the largest block of input nodes (that is, nodes with
        ``_in_degree`` equal to 0) that match a given filtering function.
        Examples of this include collecting blocks of swap gates,
        blocks of linear gates (CXs and SWAPs), blocks of Clifford gates, blocks of single-qubit gates,
        blocks of two-qubit gates, etc.  Here 'iteratively' means that once a node is collected,
        the ``_in_degree`` of each of its immediate successor is decreased by 1, allowing more nodes
        to become input and to be eligible for collecting into the current block.
        Returns the block of collected nodes.
        """
        current_block = []
        unprocessed_pending_nodes = self._pending_nodes
        self._pending_nodes = []
        while unprocessed_pending_nodes:
            new_pending_nodes = []
            for node in unprocessed_pending_nodes:
                if filter_fn(node):
                    current_block.append(node)
                    for suc in self._direct_succs(node):
                        self._in_degree[suc] -= 1
                        if self._in_degree[suc] == 0:
                            new_pending_nodes.append(suc)
                else:
                    self._pending_nodes.append(node)
            unprocessed_pending_nodes = new_pending_nodes
        return current_block

    def collect_all_matching_blocks(self, filter_fn, split_blocks=True, min_block_size=2, split_layers=False, collect_from_back=False):
        """Collects all blocks that match a given filtering function filter_fn.
        This iteratively finds the largest block that does not match filter_fn,
        then the largest block that matches filter_fn, and so on, until no more uncollected
        nodes remain. Intuitively, finding larger blocks of non-matching nodes helps to
        find larger blocks of matching nodes later on.

        After the blocks are collected, they can be optionally refined. The option
        ``split_blocks`` allows to split collected blocks into sub-blocks over disjoint
        qubit subsets. The option ``split_layers`` allows to split collected blocks
        into layers of non-overlapping instructions. The option ``min_block_size``
        specifies the minimum number of gates in the block for the block to be collected.

        By default, blocks are collected in the direction from the inputs towards the outputs
        of the circuit. The option ``collect_from_back`` allows to change this direction,
        that is collect blocks from the outputs towards the inputs of the circuit.

        Returns the list of matching blocks only.
        """

        def not_filter_fn(node):
            """Returns the opposite of filter_fn."""
            return not filter_fn(node)
        self._collect_from_back = collect_from_back
        self._setup_in_degrees()
        matching_blocks = []
        while self._have_uncollected_nodes():
            self.collect_matching_block(not_filter_fn)
            matching_block = self.collect_matching_block(filter_fn)
            if matching_block:
                matching_blocks.append(matching_block)
        if split_layers:
            tmp_blocks = []
            for block in matching_blocks:
                tmp_blocks.extend(split_block_into_layers(block))
            matching_blocks = tmp_blocks
        if split_blocks:
            tmp_blocks = []
            for block in matching_blocks:
                tmp_blocks.extend(BlockSplitter().run(block))
            matching_blocks = tmp_blocks
        if self._collect_from_back:
            matching_blocks = [block[::-1] for block in matching_blocks[::-1]]
        matching_blocks = [block for block in matching_blocks if len(block) >= min_block_size]
        return matching_blocks