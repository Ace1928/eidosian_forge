from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit import (
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
Run the pass on one :class:`.DAGCircuit`, mutating it.  Returns ``True`` if the circuit
        was modified and ``False`` if not.