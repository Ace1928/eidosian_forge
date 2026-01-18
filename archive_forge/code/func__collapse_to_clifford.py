from functools import partial
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
from qiskit.quantum_info.operators import Clifford
def _collapse_to_clifford(circuit):
    """Specifies how to construct a ``Clifford`` from a quantum circuit (that must
    consist of Clifford gates only)."""
    return Clifford(circuit)