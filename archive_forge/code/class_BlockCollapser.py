from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
class BlockCollapser:
    """This class implements various strategies of consolidating blocks of nodes
     in a DAG (direct acyclic graph). It works both with the
    :class:`~qiskit.dagcircuit.DAGCircuit` and
    :class:`~qiskit.dagcircuit.DAGDependency` DAG representations.
    """

    def __init__(self, dag):
        """
        Args:
            dag (Union[DAGCircuit, DAGDependency]): The input DAG.
        """
        self.dag = dag

    def collapse_to_operation(self, blocks, collapse_fn):
        """For each block, constructs a quantum circuit containing instructions in the block,
        then uses collapse_fn to collapse this circuit into a single operation.
        """
        global_index_map = {wire: idx for idx, wire in enumerate(self.dag.qubits)}
        global_index_map.update({wire: idx for idx, wire in enumerate(self.dag.clbits)})
        for block in blocks:
            cur_qubits = set()
            cur_clbits = set()
            cur_clregs = set()
            for node in block:
                cur_qubits.update(node.qargs)
                cur_clbits.update(node.cargs)
                cond = getattr(node.op, 'condition', None)
                if cond is not None:
                    cur_clbits.update(condition_resources(cond).clbits)
                    if isinstance(cond[0], ClassicalRegister):
                        cur_clregs.add(cond[0])
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map[x])
            sorted_clbits = sorted(cur_clbits, key=lambda x: global_index_map[x])
            qc = QuantumCircuit(sorted_qubits, sorted_clbits)
            for reg in cur_clregs:
                qc.add_register(reg)
            wire_pos_map = {qb: ix for ix, qb in enumerate(sorted_qubits)}
            wire_pos_map.update({qb: ix for ix, qb in enumerate(sorted_clbits)})
            for node in block:
                instructions = qc.append(CircuitInstruction(node.op, node.qargs, node.cargs))
                cond = getattr(node.op, 'condition', None)
                if cond is not None:
                    instructions.c_if(*cond)
            op = collapse_fn(qc)
            self.dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)
        return self.dag