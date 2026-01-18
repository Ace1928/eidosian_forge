import logging
import math
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.library.standard_gates import (
from qiskit.circuit import Qubit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
def _gate_sequence_to_dag(self, best_synth_circuit):
    qubits = (Qubit(),)
    out_dag = DAGCircuit()
    out_dag.add_qubits(qubits)
    out_dag.global_phase = best_synth_circuit.global_phase
    for gate_name, angles in best_synth_circuit:
        out_dag.apply_operation_back(NAME_MAP[gate_name](*angles), qubits, check=False)
    return out_dag