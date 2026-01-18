from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils
def _fake_run(self, dag):
    """Do a fake run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to improve initial layout.

        Returns:
            DAGCircuit: The same DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
    if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
        raise TranspilerError('Basic swap runs on physical circuits only')
    if len(dag.qubits) > len(self.coupling_map.physical_qubits):
        raise TranspilerError('The layout does not match the amount of qubits in the DAG')
    canonical_register = dag.qregs['q']
    trivial_layout = Layout.generate_trivial_layout(canonical_register)
    current_layout = trivial_layout.copy()
    for layer in dag.serial_layers():
        subdag = layer['graph']
        for gate in subdag.two_qubit_ops():
            physical_q0 = current_layout[gate.qargs[0]]
            physical_q1 = current_layout[gate.qargs[1]]
            if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                for swap in range(len(path) - 2):
                    current_layout.swap(path[swap], path[swap + 1])
    self.property_set['final_layout'] = current_layout
    return dag