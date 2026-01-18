from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumRegister, ControlledGate, Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import CZGate, CU1Gate, MCU1Gate
from qiskit.utils import optionals as _optionals
def _multigate_opt(self, dag, qubit, max_idx=None, dnt_rec=None):
    """
        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.
            qubit (Qubit): qubit whose gate cache is to be optimized
            max_idx (int): a value indicates a recursive call, optimize
                           and remove gates up to this point in the cache
            dnt_rec (list(int)): don't recurse on these qubit caches (again)
        """
    if not self.gatecache[qubit]:
        return
    self._remove_successive_identity(dag, qubit, max_idx)
    if len(self.gatecache[qubit]) < self.size and max_idx is None:
        return
    elif max_idx is None:
        max_idx = 0
        dnt_rec = set()
        dnt_rec.add(qubit)
        gates_tbr = [self.gatecache[qubit][0]]
    else:
        gates_tbr = self.gatecache[qubit][max_idx::-1]
    for node in gates_tbr:
        new_qb = [x for x in node.qargs if x not in dnt_rec]
        dnt_rec.update(new_qb)
        for qbt in new_qb:
            idx = self.gatecache[qbt].index(node)
            self._multigate_opt(dag, qbt, max_idx=idx, dnt_rec=dnt_rec)
    self.gatecache[qubit] = self.gatecache[qubit][max_idx + 1:]