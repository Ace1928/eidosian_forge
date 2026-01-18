from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import AnalysisPass
def _visit_target(dag, wire_map):
    for gate in dag.op_nodes():
        if gate.name == 'barrier':
            continue
        if not self._target.instruction_supported(gate.name, tuple((wire_map[bit] for bit in gate.qargs))):
            return True
        if isinstance(gate.op, ControlFlowOp):
            for block in gate.op.blocks:
                inner_wire_map = {inner: wire_map[outer] for outer, inner in zip(gate.qargs, block.qubits)}
                if _visit_target(circuit_to_dag(block), inner_wire_map):
                    return True
    return False