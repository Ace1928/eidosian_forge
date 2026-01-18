from typing import Type, Union, List, Optional
from fnmatch import fnmatch
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.circuit.gate import Gate
def _should_decompose(self, node) -> bool:
    """Call a decomposition pass on this circuit,
        to decompose one level (shallow decompose)."""
    if self.gates_to_decompose is None:
        return True
    if not isinstance(self.gates_to_decompose, list):
        gates = [self.gates_to_decompose]
    else:
        gates = self.gates_to_decompose
    strings_list = [s for s in gates if isinstance(s, str)]
    gate_type_list = [g for g in gates if isinstance(g, type)]
    if getattr(node.op, 'label', None) is not None and node.op.label != '' and (node.op.label in gates or any((fnmatch(node.op.label, p) for p in strings_list))):
        return True
    elif node.name in gates or any((fnmatch(node.name, p) for p in strings_list)):
        return True
    elif any((isinstance(node.op, op) for op in gate_type_list)):
        return True
    else:
        return False