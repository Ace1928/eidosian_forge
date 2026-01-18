from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
class EnlargeWithAncilla(TransformationPass):
    """Extend the dag with virtual qubits that are in layout but not in the circuit yet.

    Extend the DAG circuit with new virtual qubits (ancilla) that are specified
    in the layout, but not present in the circuit. Which qubits to add are
    previously allocated in the ``layout`` property, by a previous pass.
    """

    def run(self, dag):
        """Run the EnlargeWithAncilla pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to extend.

        Returns:
            DAGCircuit: An extended DAG.

        Raises:
            TranspilerError: If there is no layout in the property set or not set at init time.
        """
        layout = self.property_set['layout']
        if layout is None:
            raise TranspilerError('EnlargeWithAncilla requires property_set["layout"]')
        new_qregs = {reg for reg in layout.get_registers() if reg not in dag.qregs.values()}
        for qreg in new_qregs:
            dag.add_qreg(qreg)
        return dag