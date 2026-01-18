from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals
def _remove_control(self, gate, ctrlvar, trgtvar):
    """use z3 sat solver to determine if all control qubits are in 1 state,
             and if so replace the Controlled - U by U.
        Args:
            gate (Gate): gate to inspect
            ctrlvar (list(BoolRef)): z3 variables corresponding to latest state
                                     of control qubits
            trgtvar (list(BoolRef)): z3 variables corresponding to latest state
                                     of target qubits
        Returns:
            Tuple(bool, DAGCircuit, List)::
              * bool:if controlled gate can be replaced.
              * DAGCircuit: with U applied to the target qubits.
              * List: with indices of target qubits.
        """
    remove = False
    qarg = QuantumRegister(gate.num_qubits)
    dag = DAGCircuit()
    dag.add_qreg(qarg)
    qb = list(range(len(ctrlvar), gate.num_qubits))
    if isinstance(gate, ControlledGate):
        remove = self._check_removal(ctrlvar)
    if isinstance(gate, (CZGate, CU1Gate, MCU1Gate)):
        while not remove and qb[0] > 0:
            qb[0] = qb[0] - 1
            ctrl_vars = ctrlvar[:qb[0]] + ctrlvar[qb[0] + 1:] + trgtvar
            remove = self._check_removal(ctrl_vars)
    if remove:
        qubits = [qarg[qi] for qi in qb]
        dag.apply_operation_back(gate.base_gate, qubits)
    return (remove, dag, qb)