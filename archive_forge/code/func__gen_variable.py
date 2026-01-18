from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals
def _gen_variable(self, qubit):
    """After each gate generate a new unique variable name for each of the
            qubits, using scheme: 'q[id]_[gatenum]', e.g. q1_0 -> q1_1 -> q1_2,
                                                          q2_0 -> q2_1
        Args:
            qubit (Qubit): qubit to generate new variable for
        Returns:
            BoolRef: z3 variable of qubit state
        """
    import z3
    varname = 'q' + str(qubit) + '_' + str(self.gatenum[qubit])
    var = z3.Bool(varname)
    self.gatenum[qubit] += 1
    self.variables[qubit].append(var)
    return var