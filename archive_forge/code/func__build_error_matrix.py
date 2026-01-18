import numpy as np
import rustworkx
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit._accelerate.dense_layout import best_subset
def _build_error_matrix(num_qubits, qubit_map, target=None, coupling_map=None, backend_prop=None):
    error_mat = np.zeros((num_qubits, num_qubits))
    use_error = False
    if target is not None and target.qargs is not None:
        for qargs in target.qargs:
            if len(qargs) > 2:
                continue
            error = 0.0
            ops = target.operation_names_for_qargs(qargs)
            for op in ops:
                props = target[op].get(qargs, None)
                if props is not None and props.error is not None:
                    error = max(error, props.error)
            max_error = error
            if any((qubit not in qubit_map for qubit in qargs)):
                continue
            if len(qargs) == 1:
                qubit = qubit_map[qargs[0]]
                error_mat[qubit][qubit] = max_error
                use_error = True
            elif len(qargs) == 2:
                error_mat[qubit_map[qargs[0]]][qubit_map[qargs[1]]] = max_error
                use_error = True
    elif backend_prop and coupling_map:
        error_dict = {tuple(gate.qubits): gate.parameters[0].value for gate in backend_prop.gates if len(gate.qubits) == 2}
        for edge in coupling_map.get_edges():
            gate_error = error_dict.get(edge)
            if gate_error is not None:
                if edge[0] not in qubit_map or edge[1] not in qubit_map:
                    continue
                error_mat[qubit_map[edge[0]]][qubit_map[edge[1]]] = gate_error
                use_error = True
        for index, qubit_data in enumerate(backend_prop.qubits):
            if index not in qubit_map:
                continue
            for item in qubit_data:
                if item.name == 'readout_error':
                    mapped_index = qubit_map[index]
                    error_mat[mapped_index][mapped_index] = item.value
                    use_error = True
    return (error_mat, use_error)