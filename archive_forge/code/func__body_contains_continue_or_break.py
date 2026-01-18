from qiskit.circuit import ForLoopOp, ContinueLoopOp, BreakLoopOp, IfElseOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag
def _body_contains_continue_or_break(circuit):
    """Checks if a circuit contains ``continue``s or ``break``s. Conditional bodies are inspected."""
    for inst in circuit.data:
        operation = inst.operation
        if isinstance(operation, (ContinueLoopOp, BreakLoopOp)):
            return True
        if isinstance(operation, IfElseOp):
            for block in operation.params:
                if _body_contains_continue_or_break(block):
                    return True
    return False