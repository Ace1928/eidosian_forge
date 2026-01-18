from qiskit.circuit import QuantumCircuit, CircuitInstruction
Build a ``QuantumCircuit`` object from a ``DAGDependency``.

    Args:
        dagdependency (DAGDependency): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag dependency.
    