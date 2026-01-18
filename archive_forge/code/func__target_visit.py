from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import AnalysisPass
def _target_visit(self, dag, wire_map):
    for node in dag.op_nodes(include_directives=False):
        if isinstance(node.op, ControlFlowOp):
            for block in node.op.blocks:
                inner_wire_map = {inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)}
                if not self._target_visit(circuit_to_dag(block), inner_wire_map):
                    return False
        elif len(node.qargs) == 2 and (not self.target.instruction_supported(node.op.name, (wire_map[node.qargs[0]], wire_map[node.qargs[1]]))):
            return False
    return True