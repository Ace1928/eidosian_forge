import logging
import math
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.library.standard_gates import (
from qiskit.circuit import Qubit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
def _build_error_map(self):
    if self._target is not None and self._target.num_qubits is not None:
        error_map = euler_one_qubit_decomposer.OneQubitGateErrorMap(self._target.num_qubits)
        for qubit in range(self._target.num_qubits):
            gate_error = {}
            for gate, gate_props in self._target.items():
                if gate_props is not None:
                    props = gate_props.get((qubit,), None)
                    if props is not None and props.error is not None:
                        gate_error[gate] = props.error
            error_map.add_qubit(gate_error)
        return error_map
    else:
        return None