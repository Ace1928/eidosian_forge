from functools import partial
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
def _is_linear_gate(node):
    """Specifies whether a node holds a linear gate."""
    return node.op.name in ('cx', 'swap') and getattr(node.op, 'condition', None) is None