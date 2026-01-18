from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals
def _test_gate(self, gate, ctrl_ones, trgtvar):
    """use z3 sat solver to determine triviality of gate
        Args:
            gate (Gate): gate to inspect
            ctrl_ones (BoolRef): z3 condition asserting all control qubits to 1
            trgtvar (list(BoolRef)): z3 variables corresponding to latest state
                                     of target qubits
        Returns:
            bool: if gate is trivial
        """
    import z3
    trivial = False
    self.solver.push()
    try:
        triv_cond = gate._trivial_if(*trgtvar)
    except AttributeError:
        self.solver.add(ctrl_ones)
        trivial = self.solver.check() == z3.unsat
    else:
        if isinstance(triv_cond, bool):
            if triv_cond and len(trgtvar) == 1:
                self.solver.add(z3.Not(z3.And(ctrl_ones, trgtvar[0])))
                sol1 = self.solver.check() == z3.unsat
                self.solver.pop()
                self.solver.push()
                self.solver.add(z3.And(ctrl_ones, trgtvar[0]))
                sol2 = self.solver.check() == z3.unsat
                trivial = sol1 or sol2
        else:
            self.solver.add(z3.And(ctrl_ones, z3.Not(triv_cond)))
            trivial = self.solver.check() == z3.unsat
    self.solver.pop()
    return trivial