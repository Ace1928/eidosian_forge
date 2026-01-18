from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals
def _check_removal(self, ctrlvar):
    import z3
    ctrl_ones = z3.And(*ctrlvar)
    self.solver.push()
    self.solver.add(z3.Not(ctrl_ones))
    remove = self.solver.check() == z3.unsat
    self.solver.pop()
    return remove